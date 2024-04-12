from pytorch_lightning.loggers import NeptuneLogger

class MyNeptuneLogger(NeptuneLogger):

    def __init__(self):
        super().__init__(api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZDM3MmY2MS0xYjNhLTRmODEtOGQxMi1jYTkzZTMwYTQwMjcifQ==",
                         project="derolol/tunnel-gen-prototype",
                         tags=["tunnel", "center_work"])