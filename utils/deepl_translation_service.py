import boto3
import awswrangler.secretsmanager as sm
import boto3
from dataclasses import dataclass
import deepl
from loguru import logger


@dataclass
class SecretsManagerBlueprint:
    secret_id: str = "ds-llm-production"
    region_name: str = "eu-central-1"
    profile_name: str = "739275445236_DataScienceUser"

def get_secret(secret_id: str, value: str, aws_region: str, profile_name: str = None) -> str:
    session_kwargs = {"region_name": aws_region}
    if profile_name:
        session_kwargs["profile_name"] = profile_name
    return sm.get_secret_json(
        secret_id,
        boto3_session=boto3.Session(**session_kwargs),
    ).get(value)


class TranslationService:
    def __init__(self):
        self.secrets_manager = SecretsManagerBlueprint()
        self._translator = deepl.Translator(get_secret(
            self.secrets_manager.secret_id,
            "deepl_token",
            self.secrets_manager.region_name,
            self.secrets_manager.profile_name
        )
        )

    def translate(self, text, target_lang="EN-GB"):
        try:
            return self._translator.translate_text(text, target_lang=target_lang).text
            # return text
        except Exception as e:
            logger.error(f'Error in translate text: {text}')
            logger.opt(exception=True).error(str(e))
            return text