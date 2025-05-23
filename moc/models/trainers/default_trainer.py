class DefaultTrainer:
    def __init__(self, rc, **kwargs):
        self.rc = rc

    def fit(self, model, datamodule):
        self.model = model
        self.datamodule = datamodule
        return model.fit(datamodule)

    def test(self, model, datamodule, **kwargs):
        return model.test(datamodule)
