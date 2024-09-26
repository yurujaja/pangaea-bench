from collections import OrderedDict
from typing import List, Dict, Union, Tuple
import torch


SENSORS = ["s2", "s1", "l8", "l7", "l5", "l4"]
SENSOR_BANDS = {
    "s2": ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b8a", "b9", "b11", "b12"],
    "s1": ["vv", "vh"],
    "l8": ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11", "b12"],
    "l7": ["b1", "b2", "b3", "b4", "b5", "b7"],
    "l5": ["b1", "b2", "b3", "b4", "b5", "b6", "b7"],
    "l4": ["b1", "b2", "b3", "b4", "b5", "b6", "b7"],
}


class SensorData(OrderedDict):
    """
    A class to represent sensor data for a specific sensor. This class extends OrderedDict
    to store band data as key-value pairs where the key is the band name and the value is
    a torch.Tensor representing the band data.

    Attributes:
        sensor (str): The name of the sensor.
        bands (List[str]): The list of bands for the sensor.
    """

    def __init__(self, sensor: str) -> None:
        """
        Initializes the SensorData object with the given sensor name.

        Args:
            sensor (str): The name of the sensor.
        """
        super(SensorData, self).__init__()
        self.sensor = sensor
        self.bands = SENSOR_BANDS[sensor]

    def __setitem__(self, band_name: str, band_data: torch.Tensor) -> None:
        """
        Sets the band data for a specific band name. If the band name is "all", it sets
        the data for all bands at once.

        Args:
            band_name (str): The name of the band.
            band_data (torch.Tensor): The data for the band.

        Raises:
            ValueError: If the band name is not a string or the band data is not a torch.Tensor.
            ValueError: If the band name is "all" and the shape of the band data does not match
                        the expected number of bands.
            ValueError: If the band name is not in the list of valid bands for the sensor.
        """
        if not isinstance(band_name, str):
            raise ValueError(f"Band name must be a string, got {type(band_name)}")
        if not isinstance(band_data, torch.Tensor):
            raise ValueError(f"Value must be a torch.Tensor, got {type(band_data)}")

        if band_name == "all":
            # all bands are provided all at once
            assert (
                len(band_data.shape) == 3
            ), f"Expected 3D tensor, got {band_data.shape}"
            if band_data.shape[-3] != len(self.bands):
                raise ValueError(
                    f"Expected {len(self.bands)} bands, got {band_data.shape[-3]}"
                )
            for bn, bd in zip(self.bands, band_data):
                self.__setitem__(bn, bd)
        else:
            assert (
                len(band_data.shape) == 2
            ), f"Expected 2D tensor, got {band_data.shape}"
            if band_name not in self.bands:
                raise ValueError(f"Invalid band name: {band_name}")
            super(SensorData, self).__setitem__(band_name, band_data)

    def __getitem__(self, band_name: str) -> torch.Tensor:
        """
        Gets the band data for a specific band name. If the band name is "all", it returns
        the data for all bands stacked along the first dimension.

        Args:
            band_name (str): The name of the band.

        Raises:
            ValueError: If the band name is not a string.

        Returns:
            torch.Tensor: The data for the specified band or all bands.
        """
        if not isinstance(band_name, str):
            raise ValueError(f"Band name must be a string, got {type(band_name)}")

        if band_name == "all":
            band_data = []
            for bn in self.bands:
                if bn not in self.keys():
                    # if band not in self.keys():
                    # we pad the missing band with zeros
                    band_data.append(torch.zeros_like(list(self.values())[0]))
                else:
                    band_data.append(self[bn])
            return torch.stack(band_data, dim=-3)

        else:
            if band_name not in self.keys():
                return None
            return super(SensorData, self).__getitem__(band_name)

    def to(self, device):
        """
        Moves all band data to the specified device.

        Args:
            device: The device to move the data to.

        Returns:
            SensorData: The SensorData object with data moved to the specified device.
        """
        for key, value in self.items():
            self.__setitem__(key, value.to(device))
        return self

    def to_dtype(self, dtype):
        """
        Converts all band data to the specified dtype.

        Args:
            dtype: The dtype to convert the data to.

        Returns:
            SensorData: The SensorData object with data converted to the specified dtype.
        """
        for key, value in self.items():
            self.__setitem__(key, value.to(dtype))
        return self

    def to_device_dtype(self, device, dtype):
        """
        Moves all band data to the specified device and converts it to the specified dtype.

        Args:
            device: The device to move the data to.
            dtype: The dtype to convert the data to.

        Returns:
            SensorData: The SensorData object with data moved to the specified device and
                        converted to the specified dtype.
        """
        for key, value in self.items():
            self.__setitem__(key, value.to(device, dtype))
        return self

class TimeSerieData(OrderedDict):
    def __init__(self, sensor: str) -> None:
        super(TimeSerieData, self).__init__()
        self.sensor = sensor
        self.bands = SENSOR_BANDS[sensor]

    def __setitem__(self, band_name: str, timeserie: torch.Tensor) -> None:
        # check if key is a string
        if not isinstance(band_name, str):
            raise ValueError(f"Band name must be a string, got {type(band_name)}")
        if not isinstance(timeserie, torch.Tensor):
            raise ValueError(f"Value must be a torch.Tensor, got {type(timeserie)}")

        assert len(timeserie.shape) in [
            3,
            4,
        ], f"Expected 3D or 4D tensor, got {timeserie.shape}"
        timeserie_data = []
        for band_data in timeserie:
            sd = SensorData(self.sensor)
            sd[band_name] = band_data
            timeserie_data.append(sd)

        super(TimeSerieData, self).__setitem__(band_name, timeserie_data)

    def __getitem__(self, band_name: str) -> torch.Tensor:
        if not isinstance(band_name, str):
            raise ValueError(f"Band name must be a string, got {type(band_name)}")

        if band_name == "all":
            # get the first key
            k = list(self.keys())[0]
            sensor_data = super(TimeSerieData, self).__getitem__(k)
        elif band_name not in self.keys():
            return None
        else:
            sensor_data = super(TimeSerieData, self).__getitem__(band_name)
        sensor_data = [sd[band_name] for sd in sensor_data]

        return torch.stack(sensor_data, dim=0)

    def to(self, device):
        for key, value in self.items():
            timeserie = [data.to(device) for data in value]
            super(TimeSerieData, self).__setitem__(key, timeserie)
        return self

    def to_dtype(self, dtype):
        for key, value in self.items():
            self.__setitem__(key, value.to(dtype))
        return self

    def to_device_dtype(self, device, dtype):
        for key, value in self.items():
            self.__setitem__(key, value.to(device, dtype))
        return self


class EoTensor(OrderedDict):
    """
    A class to represent Earth Observation data.
    """

    def __init__(self, *args, **kwargs) -> None:
        super(EoTensor, self).__init__(*args, **kwargs)
        self.sensor_data = OrderedDict()
        self.timeseries = OrderedDict()

    def __parse_key(self, key: str) -> Tuple[str, str | None, bool]:
        # the format of the key is "sensor - [band] - [timeserie]" with the following rules:
        # - sensor: one of the supported sensors
        # - band: one of the bands of the sensor (optional)
        # - timeserie: one of the timeseries of the sensor (optional)
        # remove leading and trailing whitespaces
        key_parts = [kp.strip() for kp in key.split("-")]
        sensor = key_parts[0]
        assert sensor in SENSORS, f"Invalid sensor name: {sensor}"
        if len(key_parts) == 3 and key_parts[2] == "ts":
            band = key_parts[1]
            timeserie = True
        elif len(key_parts) == 2:
            # we need to check if the second part is a band or a timeserie
            if key_parts[1] == "ts":
                timeserie = True
                band = "all"
            else:
                timeserie = False
                band = key_parts[1]
                assert (
                    band in SENSOR_BANDS[sensor]
                ), f"Invalid band name: {band} for sensor {sensor}"
        else:
            band = "all"
            timeserie = False

        return sensor, band, timeserie

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        # check if key is a string
        if not isinstance(key, str):
            raise ValueError(f"Key must be a string, got {type(key)}")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Value must be a torch.Tensor, got {type(value)}")

        sensor, band, timeserie = self.__parse_key(key)
        if timeserie:
            self.__set_timeserie(sensor, band, value)
        else:
            self.__set_sensordata(sensor, band, value)

    def __set_sensordata(self, sensor: str, band: str, value: torch.Tensor) -> None:
        sd = SensorData(sensor)
        sd[band] = value
        self.sensor_data[sensor] = sd

    def __set_timeserie(self, sensor: str, band: str, value: torch.Tensor) -> None:
        ts = TimeSerieData(sensor)
        ts[band] = value
        self.timeseries[sensor] = ts

    def __getitem__(self, key: str) -> torch.Tensor:
        # check if key is a string
        if not isinstance(key, str):
            raise ValueError(f"Key must be a string, got {type(key)}")

        sensor, band, timeserie = self.__parse_key(key)
        if timeserie:
            return self.__get_timeserie(sensor, band)
        else:
            return self.__get_sensordata(sensor, band)

    def __get_sensordata(self, sensor: str, band: str) -> torch.Tensor:
        sd = self.sensor_data[sensor]
        return sd[band]

    def __get_timeserie(self, sensor: str, band: str) -> torch.Tensor:
        ts = self.timeseries[sensor]
        return ts[band]

    def __repr__(self):
        return f"EoTensor({super(EoTensor, self).__repr__()})"

    def __str__(self):
        return f"EoTensor({super(EoTensor, self).__str__()})"

    def __len__(self):
        return super(EoTensor, self).__len__()

    def __iter__(self):
        return super(EoTensor, self).__iter__()

    def __contains__(self, key):
        return super(EoTensor, self).__contains__(key)

    def keys(self):
        return super(EoTensor, self).keys()

    def values(self):
        return super(EoTensor, self).values()

    def items(self):
        return super(EoTensor, self).items()

    def to(self, device):
        for key, value in self.sensor_data.items():
            self.sensor_data[key] = value.to(device)
        for key, value in self.timeseries.items():
            self.timeseries[key] = value.to(device)
        return self

    def to_dtype(self, dtype):
        for key, value in self.items():
            self.__setitem__(key, value.to(dtype))
        return self

    def to_device_dtype(self, device, dtype):
        for key, value in self.items():
            self.__setitem__(key, value.to(device, dtype))
        return self

    def to_dict(self):
        return dict(self.items())

    def to_list(self):
        return list(self.values())

    def to_tuple(self):
        return tuple(self.values())

    def to_dict_list(self):
        return list(self.items())

    def to_dict_tuple(self):
        return tuple(self.items())

    def to_dict_dict(self):
        return dict(self.items())

    def to_dict_dict_list(self):
        return [dict]


if __name__ == "__main__":
    # print("Only one band")
    # band = torch.rand(256, 256)
    # s2 = SensorData("s2")
    # s2["b1"] = band
    # print(s2["b1"].shape)
    #
    # print("All bands")
    # s2 = SensorData("s2")
    # s2_data = torch.rand(12, 256, 256)
    # s2["all"] = s2_data
    # print(s2["b1"].shape)
    # print(s2["all"].shape)
    #
    # print("Set two bands, get all bands")
    # s2 = SensorData("s2")
    # b1 = torch.rand(256, 256)
    # b2 = torch.rand(256, 256)
    # s2["b1"] = b1
    # s2["b2"] = b2
    # print(s2["b1"].shape)
    # print(s2["all"].shape)

    print("One band")
    t = EoTensor()
    t["s2-b1"] = torch.rand(256, 256)
    print(t["s2-b1"].shape)

    print("All bands")
    t = EoTensor()
    t["s2"] = torch.rand(12, 256, 256)
    print(t["s2"].shape)

    print("One band timeserie")
    t = EoTensor()
    t["s2-b1-ts"] = torch.rand(3, 256, 256)
    x = t["s2-b1-ts"]
    print(x.shape)

    print("All bands timeserie")
    t = EoTensor()
    t["s2-ts"] = torch.rand(15, 12, 256, 256)
    print("timeserie s2 : ", t["s2-ts"].shape)
    t["s2"] = torch.rand(12, 32, 32)
    print("s2 : ", t["s2"].shape)

    print("CHECK DEVICE")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t.to(device)
    print("timeserie s2 : ", t["s2-ts"].device)
    print("s2 : ", t["s2"].device)

    # ts = TimeSerieData("s2")
    # ts["all"] = torch.rand(9, 12, 256, 256)
    # print(ts["all"].shape)
    #
    # x = torch.rand(9, 256, 256)
    # print("DEvice test", x.device)
    # ts = TimeSerieData("s2")
    # ts["b1"] = torch.rand(9, 256, 256)
    #
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print("Device:", device)
    # ts.to(device)
    #
    # print(ts["all"].shape)
    # print(ts["all"].device)
    #
    # print(ts["b1"].shape)
    # print(ts["b2"])
