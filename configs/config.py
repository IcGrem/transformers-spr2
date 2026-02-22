from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    SEQ_LEN: int = 3
    BATCH_SIZE: int = 256
    EMBEDDING_DIM: int = 128
    HIDDEN_DIM: int = 512
    NUM_LAYERS: int = 1
    DROPOUT: float = 0.3
    EPOCH: int = 5
    MAX_ROUGE_SAMPLES: int = 10
    MAX_GEN_LENGTH: int = 300
    
    TEMPERATURE: float = 0.8
    TOP_K: int = 50

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings() # type: ignore
