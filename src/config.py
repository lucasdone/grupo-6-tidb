"""
FerryFlow – Configuration

Loads all environment variables. Never logs credentials.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file if present (local development)
load_dotenv()


@dataclass
class Config:
    # Amazon Bedrock
    aws_bearer_token_bedrock: str
    aws_region: str

    # TiDB Cloud
    tidb_host: str
    tidb_port: int
    tidb_user: str
    tidb_password: str
    tidb_database: str
    tidb_ssl_ca: str

    # Derived flags
    @property
    def bedrock_configured(self) -> bool:
        return bool(self.aws_bearer_token_bedrock and self.aws_region)

    @property
    def tidb_configured(self) -> bool:
        return bool(self.tidb_host and self.tidb_user and self.tidb_password)


def load_config() -> Config:
    """Load and return application configuration from environment variables."""
    return Config(
        aws_bearer_token_bedrock=os.getenv("AWS_BEARER_TOKEN_BEDROCK", ""),
        aws_region=os.getenv("AWS_REGION", "ap-southeast-1"),
        tidb_host=os.getenv("TIDB_HOST", ""),
        tidb_port=int(os.getenv("TIDB_PORT", "4000")),
        tidb_user=os.getenv("TIDB_USER", ""),
        tidb_password=os.getenv("TIDB_PASSWORD", ""),
        tidb_database=os.getenv("TIDB_DATABASE", "airportdb"),
        tidb_ssl_ca=os.getenv("TIDB_SSL_CA", ""),
    )


# Singleton config instance
_config: Config | None = None


def get_config() -> Config:
    """Return cached config singleton."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
