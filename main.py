from data import G2PDataModule, InflectionDataModule
from pytorch_lightning import Trainer
from model import InterpretableTransducer
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping


if __name__ == '__main__':
    name = "german_g2p"
    logger = pl_loggers.CSVLogger(save_dir="./logs", name=name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./saved_models/{name}", filename=name + "-{val_edit_distance}", monitor="val_edit_distance",
        save_last=True, save_top_k=1, mode="min", verbose=True
    )
    early_stopping_callback = EarlyStopping(monitor="val_edit_distance", patience=3, mode="min", verbose=True)

    """
    dm = G2PDataModule.from_files(
        train_path="data/g2p_2022/german/ger_train.tsv",
        dev_path="data/g2p_2022/german/ger_dev.tsv",
        test_path="data/g2p_2022/german/ger_test.tsv.private"
    )
    """
    dm = InflectionDataModule.from_files(
        train_path="data/inflection/norse_old/non_large.train",
        dev_path="data/inflection/norse_old/non.dev",
        test_path="data/inflection/norse_old/non.gold"
    )
    # """

    dm.prepare_data()
    dm.setup(stage="fit")

    model = InterpretableTransducer(
        source_alphabet_size=dm.source_alphabet_size, target_alphabet_size=dm.target_alphabet_size,
        num_layers=2, dropout=0.0, hidden_size=256, max_decoding_length=40,
        num_source_features=10, num_symbol_features=0, num_decoder_states=10, autoregressive_order=2,
        enable_seq2seq_loss=True, binary_attention=False
    )
    trainer = Trainer(
        accelerator="gpu", gradient_clip_val=1.0,
        max_epochs=500, enable_progress_bar=True, gpus=1, log_every_n_steps=10, logger=logger,
        check_val_every_n_epoch=1, callbacks=[early_stopping_callback, checkpoint_callback]
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)

