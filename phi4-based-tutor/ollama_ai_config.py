from dotenv import load_dotenv
from os import getenv

class OllamaAIConfig:

    def __get_config_value(self, key: str) -> str:
        """
        Get the value of an environment variable.
        """
        if not key:
            raise ValueError("Key must be provided")

        value: str | None = getenv(key)
        if not value:
            raise ValueError(f"Environment variable '{key}' not found")

        return value

    def _get_int_value(self, key: str) -> int:
        """
        Get an integer value from the environment variables.
        """
        value = self.__get_config_value(key)
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Environment variable '{key}' must be an integer")

    def _get_float_value(self, key: str) -> float:
        """
        Get a float value from the environment variables.
        """
        value = self.__get_config_value(key)
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Environment variable '{key}' must be a float")

    def __get_model_id(self) -> str:
        """
        Get the model ID from the environment variables.
        """
        return self.__get_config_value("MODEL_ID")

    def __get_temperature(self) -> float:
        """
        Get the temperature from the environment variables.
        """
        return self._get_float_value("TEMPERATURE")

    def __get_amount_before_summarizing(self) -> int:
        """
        Get the amount before summarizing from the environment variables.
        """
        return self._get_int_value("AMOUNT_BEFORE_SUMMARIZING")

    def __init__(self) -> None:
        load_dotenv()
        self.__model_id: str = ""
        self.__temperature: float = 0.0
        self.__amount_before_summarizing: int = 0

    @property
    def model_id(self) -> str:
        if self.__model_id == "":
            self.__model_id = self.__get_model_id()
        return self.__model_id

    @property
    def temperature(self) -> float:
        if self.__temperature == 0.0:
            self.__temperature = self.__get_temperature()
        return self.__temperature

    @property
    def amount_before_summarizing(self) -> int:
        if self.__amount_before_summarizing == 0:
            self.__amount_before_summarizing = self.__get_amount_before_summarizing()
        return self.__amount_before_summarizing
