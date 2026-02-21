from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    SEQ_LEN: int = 3
    BATCH_SIZE: int = 256
    EMBEDDING_DIM: int = 128
    HIDDEN_DIM: int = 512
    NUM_LAYERS: int = 1
    DROPOUT: float = 0.3
    
    MAX_ROUGE_SAMPLES: int = 50
    MAX_GEN_LENGTH: int = 12
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings() # type: ignore
