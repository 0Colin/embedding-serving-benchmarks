import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

from serving.huggingface.sentence_encoder import Model


class IntentClassifier:
    """Intent classifier based on semantic encoding of the text and a classifier on top."""

    def __init__(
        self,
        df_path: str,
        clf: str = "lr",
        is_cpu: bool = False,
        is_onnx: bool = False,
        is_fp16: bool = False,
    ):
        """
        Args:
            df_path: path to the classification dataset; should contain the columns ``text``
                ``label`` and ``train_or_val``.
            encoder: a ``sentence-transformer`` based sentence encoder.
            clf: type of classifier(e.g. LogisticRegression)
        """
        self._df = pd.read_csv(df_path)
        self.encoder = Model(is_cpu, is_onnx, is_fp16)
        self.le = LabelEncoder()
        self.ss = StandardScaler()
        self._clf = clf
        if self._clf == "lr":
            self._clf_params = {
                "multi_class": "multinomial",
                "penalty": "l2",
                "max_iter": 1000,
                "class_weight": "balanced",
            }
            self.clf = LogisticRegression(**self._clf_params)

    def _encode(self):
        """Encodes the text data(using ``SentenceTransformer``) and the target labels
        using ``LabelEncoder``."""
        print("encoding data ... ", end="")
        train_txt = self._df[
            self._df.train_or_val == "train"
        ].text.values.tolist()
        self._train_X = []
        for i in range(0, len(train_txt), 32):
            batch = train_txt[i : i + 32]
            self._train_X += self.encoder.get_embeddings(batch)

        self._train_y = self.le.fit_transform(
            self._df[self._df.train_or_val == "train"].label
        )

        val_txt = self._df[self._df.train_or_val == "val"].text.values.tolist()
        self._val_X = []
        for i in range(0, len(val_txt), 32):
            batch = val_txt[i : i + 32]
            self._val_X += self.encoder.get_embeddings(batch)

        self._val_y = self.le.transform(
            self._df[self._df.train_or_val == "val"].label
        )
        print("done!")
        print(
            f"train_X: {len(self._train_X)}, train_y: {len(self._train_y)}\n"
            f"val_X  : {len(self._val_X)},   val_y: {len(self._val_y)}"
        )

    def _scale(self):
        """Normalizes the features for easy convergence."""
        print("scaling features ... ", end="")
        self._train_X = self.ss.fit_transform(self._train_X)
        self._val_X = self.ss.transform(self._val_X)
        print("done!")

    def train(self):
        """Trains the classification model."""
        print("training model ... ", end="")
        self.clf.fit(self._train_X, self._train_y)
        print("done!")

    def report(self):
        return print(
            classification_report(
                self._val_y,
                self.clf.predict(self._val_X),
                zero_division=0,
                target_names=self.le.classes_.tolist(),
            )
        )
