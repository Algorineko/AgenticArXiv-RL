import os
import json
from typing import Dict, Any
from loguru import logger


class WeatherTool:
    """天气工具类"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("WEATHER_API_KEY", "demo_key")
        # 模拟数据，可以替换为真实天气API
        self.mock_data = {
            "北京": {"temperature": 22, "weather": "晴天", "humidity": 45, "wind_speed": "3级"},
            "上海": {"temperature": 25, "weather": "多云", "humidity": 65, "wind_speed": "2级"},
            "广州": {"temperature": 28, "weather": "小雨", "humidity": 80, "wind_speed": "1级"},
            "深圳": {"temperature": 27, "weather": "多云", "humidity": 75, "wind_speed": "2级"},
            "杭州": {"temperature": 23, "weather": "阴天", "humidity": 60, "wind_speed": "3级"},
            "成都": {"temperature": 20, "weather": "雾", "humidity": 85, "wind_speed": "1级"}
        }

    def get_weather(self, city: str) -> Dict[str, Any]:
        """获取天气信息

        Args:
            city: 城市名称

        Returns:
            天气信息字典
        """
        logger.info(f"获取{city}的天气信息")

        if city in self.mock_data:
            return {
                "city": city,
                **self.mock_data[city],
                "status": "success",
                "timestamp": "2025-12-29 10:00:00"
            }
        return {
            "city": city,
            "error": f"未找到{city}的天气数据",
            "status": "error",
            "suggestion": "请确认城市名称是否正确，或联系管理员添加该城市数据"
        }

    def get_all_cities(self) -> list:
        """获取所有支持的城市列表"""
        return list(self.mock_data.keys())