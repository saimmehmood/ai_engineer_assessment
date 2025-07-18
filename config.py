import os
import litellm

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    # LLM
    KNN_EMBEDDING_DIMENSION = int(os.environ.get('KNN_EMBEDDING_DIMENSION', 1536))
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    # Response Quality Improvement
    ENABLE_RESPONSE_QUALITY_IMPROVEMENT = os.environ.get("ENABLE_RESPONSE_QUALITY_IMPROVEMENT", "true").lower() == "true"

    # Langfuse
    LANGFUSE_PUBLIC_KEY = os.environ.get('LANGFUSE_PUBLIC_KEY')
    LANGFUSE_SECRET_KEY = os.environ.get('LANGFUSE_SECRET_KEY')
    LANGFUSE_HOST = os.environ.get('LANGFUSE_HOST')

    if all([LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST]):
        litellm.success_callback = ["langfuse"]
        litellm.failure_callback = ["langfuse"]
    else:
        litellm.success_callback = ["default"]
        litellm.failure_callback = ["default"]

    # Environment
    ENV_VARS = []  # List of environment variables to pass to the container/batch

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    FLASK_CONFIG = 'DEV'
    TESTING = True
    DEBUG = True


class TestingConfig(Config):
    FLASK_CONFIG = 'TEST'
    TESTING = True
    DEBUG = True


class StagingConfig(Config):
    FLASK_CONFIG = 'STAGING'
    TESTING = False
    DEBUG = False


class ProductionConfig(Config):
    FLASK_CONFIG = 'PROD'
    TESTING = False
    DEBUG = False


config = {
    'DEV': DevelopmentConfig,
    'TEST': TestingConfig,
    'STAGING': StagingConfig,
    'PROD': ProductionConfig,
}