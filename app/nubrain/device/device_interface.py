"""
EEG device interface abstraction to support multiple EEG systems (OpenBCI Cyton,
Wearable Sensing DSI-24).
"""

import time
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop


class EEGDeviceInterface(ABC):
    """Abstract interface for EEG devices."""

    @abstractmethod
    def prepare_session(self):
        pass

    @abstractmethod
    def start_stream(self):
        pass

    @abstractmethod
    def stop_stream(self):
        pass

    @abstractmethod
    def get_board_data(self) -> np.ndarray:
        pass

    @abstractmethod
    def insert_marker(self, marker: float):
        pass

    @abstractmethod
    def release_session(self):
        pass

    @abstractmethod
    def get_device_info(self) -> Dict:
        pass


class BrainFlowDevice(EEGDeviceInterface):
    """BrainFlow-based device implementation (OpenBCI Cyton)."""

    def __init__(
        self,
        board_id: int,
        params: BrainFlowInputParams,
        eeg_channel_mapping: Dict[int, str],
    ):
        self.board_id = board_id
        self.params = params
        self.board = BoardShim(board_id, params)
        self.eeg_channel_mapping = eeg_channel_mapping
        self.board_description = BoardShim.get_board_descr(board_id)

        # Update channel names from config
        eeg_channel_idxs = sorted(list(eeg_channel_mapping.keys()))
        eeg_channel_names = [eeg_channel_mapping[idx] for idx in eeg_channel_idxs]
        self.board_description["eeg_names"] = ",".join(eeg_channel_names)

    def prepare_session(self):
        self.board.prepare_session()

    def start_stream(self):
        self.board.start_stream()

    def stop_stream(self):
        self.board.stop_stream()

    def get_board_data(self) -> np.ndarray:
        return self.board.get_board_data()

    def insert_marker(self, marker: float):
        self.board.insert_marker(marker)

    def release_session(self):
        self.board.release_session()

    def get_device_info(self) -> Dict:
        # Get actual board data to determine total channels.
        test_data = self.board.get_board_data()
        if test_data.size > 0:
            n_channels_total = test_data.shape[0]
        else:
            n_channels_total = (
                len(self.board_description["eeg_channels"]) + 1
            )  # +1 for marker

        return {
            "board_description": self.board_description,
            "sampling_rate": int(self.board_description["sampling_rate"]),
            "eeg_channels": self.board_description["eeg_channels"],
            "marker_channel": self.board_description["marker_channel"],
            "n_channels_total": n_channels_total,
        }


class DSI24Device(EEGDeviceInterface):
    """DSI-24 device implementation using LSL."""

    def __init__(
        self,
        eeg_channel_mapping: Dict[int, str],
        stream_name: str = "DSI-24",
        marker_stream_name: str = "DSI24_Markers",
    ):
        self.eeg_channel_mapping = eeg_channel_mapping
        self.stream_name = stream_name
        self.marker_stream_name = marker_stream_name

        self.inlet = None
        self.marker_outlet = None
        self.marker_inlet = None  # We'll also create an inlet to read our own markers
        self.sampling_rate = None
        self.n_channels = None
        self.channel_names = []
        self.data_buffer = []
        self.timestamp_buffer = []
        self.marker_buffer = []
        self.marker_timestamp_buffer = []

    def prepare_session(self):
        """Connect to LSL streams from DSI-Streamer."""
        print(f"Looking for LSL stream: {self.stream_name}")

        # Find EEG data stream.
        streams = resolve_byprop("name", self.stream_name, minimum=1, timeout=10.0)

        if not streams:
            raise RuntimeError(
                f"Cannot find LSL stream named '{self.stream_name}'. "
                "Make sure DSI-Streamer is running and streaming data."
            )

        # Create inlet for receiving data.
        self.inlet = StreamInlet(streams[0], max_buflen=30)

        # Get stream info.
        info = self.inlet.info()
        self.sampling_rate = info.nominal_srate()
        self.n_channels = info.channel_count()

        # Get channel names from the stream info.
        ch = info.desc().child("channels").child("channel")
        for _ in range(self.n_channels):
            label = ch.child_value("label")
            if label:
                self.channel_names.append(label)
            else:
                self.channel_names.append(f"Ch{len(self.channel_names)}")
            ch = ch.next_sibling()

        print("Connected to DSI-24 stream:")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Channels: {self.n_channels}")
        print(f"  Channel names: {self.channel_names}")

        # Create marker outlet for sending markers
        marker_info = StreamInfo(
            self.marker_stream_name,
            "Markers",
            1,  # 1 channel for markers
            0,  # Irregular sampling rate
            "float32",
            "DSI24_Markers_" + str(time.time()),
        )
        self.marker_outlet = StreamOutlet(marker_info)

        # Also create a marker inlet to read our own markers (for synchronization)
        # Give it a moment to be discoverable
        time.sleep(0.5)
        marker_streams = resolve_byprop(
            "name", self.marker_stream_name, minimum=1, timeout=5.0
        )
        if marker_streams:
            self.marker_inlet = StreamInlet(marker_streams[0])

        # Initialize buffers
        self.data_buffer = []
        self.timestamp_buffer = []
        self.marker_buffer = []
        self.marker_timestamp_buffer = []

    def start_stream(self):
        """Start receiving data (DSI-Streamer should already be streaming)."""
        if self.inlet is None:
            raise RuntimeError("Session not prepared. Call prepare_session() first.")

        # Open the stream
        self.inlet.open_stream()

        # Clear any old data
        self.inlet.flush()

        # Clear buffers
        self.data_buffer = []
        self.timestamp_buffer = []
        self.marker_buffer = []
        self.marker_timestamp_buffer = []

        print("Started receiving DSI-24 data stream")

    def stop_stream(self):
        """Stop receiving data."""
        if self.inlet:
            self.inlet.close_stream()
        print("Stopped receiving DSI-24 data stream")

    def get_board_data(self) -> np.ndarray:
        """
        Get accumulated data since last call.
        Returns data in format similar to BrainFlow: [channels x samples]
        """
        if self.inlet is None:
            return np.zeros((self.n_channels + 1, 0))

        # Pull all available EEG samples.
        samples, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=1024)

        # Pull all available marker samples if we have a marker inlet
        marker_samples = []
        marker_timestamps = []
        if self.marker_inlet:
            marker_samples, marker_timestamps = self.marker_inlet.pull_chunk(
                timeout=0.0, max_samples=1024
            )

        if not samples:
            # Return empty array with correct shape.
            return np.zeros((self.n_channels + 1, 0))

        # Convert EEG samples to numpy array and transpose.
        data = np.array(samples).T  # Shape: [channels x samples]

        # Create marker channel.
        marker_channel = np.zeros((1, data.shape[1]))

        # If we have markers, try to align them with the EEG data.
        if marker_samples and marker_timestamps:
            # TODO: Simple approach: if we have any markers, place them at the beginning
            # In a more sophisticated implementation, you'd align by timestamps.
            for i, marker_val in enumerate(marker_samples):
                if i < marker_channel.shape[1]:
                    marker_channel[0, i] = marker_val[
                        0
                    ]  # marker_samples is a list of lists

        # Combine EEG data with marker channel.
        full_data = np.vstack([data, marker_channel])

        return full_data

    def insert_marker(self, marker: float):
        """Insert a marker into the stream."""
        if self.marker_outlet:
            self.marker_outlet.push_sample([marker])

    def release_session(self):
        """Clean up resources."""
        if self.inlet:
            self.inlet.close_stream()
            del self.inlet
            self.inlet = None

        if self.marker_inlet:
            self.marker_inlet.close_stream()
            del self.marker_inlet
            self.marker_inlet = None

        if self.marker_outlet:
            del self.marker_outlet
            self.marker_outlet = None

        print("DSI-24 session released")

    def get_device_info(self) -> Dict:
        """Get DSI-24 device information (compatible existing Cyton implementation)."""
        if self.inlet is None:
            raise RuntimeError("Session not prepared. Call prepare_session() first.")

        # Map DSI-24 channels to the expected format.
        eeg_channels = list(range(self.n_channels))  # All channels are EEG

        # Build channel names string.
        if self.eeg_channel_mapping:
            # Use mapping from config if provided.
            channel_names = []
            for idx in sorted(self.eeg_channel_mapping.keys()):
                if idx < len(self.channel_names):
                    channel_names.append(self.eeg_channel_mapping[idx])
                else:
                    channel_names.append(f"Ch{idx}")
            channel_names_str = ",".join(channel_names)
        else:
            channel_names_str = ",".join(self.channel_names)

        board_description = {
            "name": "DSI-24",
            "sampling_rate": self.sampling_rate,
            "eeg_channels": eeg_channels,
            "eeg_names": channel_names_str,
            "marker_channel": self.n_channels,  # Last channel is marker
        }

        return {
            "board_description": board_description,
            "sampling_rate": int(self.sampling_rate),
            "eeg_channels": eeg_channels,
            "marker_channel": self.n_channels,
            "n_channels_total": self.n_channels + 1,  # +1 for marker channel
        }


def create_eeg_device(device_type: str, **kwargs) -> EEGDeviceInterface:
    """
    Factory function to create EEG device instance.

    Args:
        device_type: 'cyton', 'synthetic', or 'dsi24'
        **kwargs: Device-specific parameters

    Returns:
        EEGDeviceInterface instance
    """
    if device_type == "dsi24":
        return DSI24Device(
            eeg_channel_mapping=kwargs.get("eeg_channel_mapping", {}),
            stream_name=kwargs.get("lsl_stream_name", "DSI-24"),
        )
    elif device_type == "synthetic":
        params = BrainFlowInputParams()
        return BrainFlowDevice(
            BoardIds.SYNTHETIC_BOARD.value,
            params,
            kwargs.get("eeg_channel_mapping", {}),
        )
    elif device_type == "cyton":
        params = BrainFlowInputParams()
        params.serial_port = kwargs.get("eeg_device_address", "")
        return BrainFlowDevice(
            BoardIds.CYTON_BOARD.value, params, kwargs.get("eeg_channel_mapping", {})
        )
    else:
        raise ValueError(f"Unknown device type: {device_type}")
