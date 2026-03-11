import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Constants / Helpers
# -----------------------------
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

ACTIVATION_MAP = {
    "relu": "relu",
    "tanh": "tanh",
    "logistic": "sigmoid",  # sklearn-style "logistic" -> Keras "sigmoid"
}


@st.cache_data(show_spinner=False)
def load_fashion_mnist(test_size=0.2, random_state=42, normalize=True):
    """
    Carga Fashion-MNIST desde OpenML usando sklearn.
    X: (n_samples, 784) float32
    y: int64 [0..9]
    """
    ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False)

    X = ds.data.astype("float32")
    y = ds.target.astype("int64")

    if normalize:
        X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def build_network(
    input_dim: int,
    num_classes: int,
    hidden_layers=(128, 64),
    activation="relu",
    dropout=0.2,
    learning_rate=1e-3,
):
    act = ACTIVATION_MAP.get(activation, activation)

    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs

    for i, units in enumerate(hidden_layers, start=1):
        x = layers.Dense(units, activation=act, name=f"dense_{i}")(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i}")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="fashion_dense_net")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_predict(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=10,
    batch_size=64,
    early_stop=True,
    patience=3,
):
    callbacks = []
    if early_stop:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        )

    h = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    history = {
        "loss": h.history.get("loss", []),
        "val_loss": h.history.get("val_loss", []),
        "accuracy": h.history.get("accuracy", []),
        "val_accuracy": h.history.get("val_accuracy", []),
    }

    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1).astype("int64")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    eval_metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_acc)}

    return history, y_pred, eval_metrics


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    return fig


def classification_report_df(y_true, y_pred, class_names):
    rpt = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(rpt).T


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fashion-MNIST: MLP vs DNN", layout="wide")

st.title("Fashion-MNIST: MLP vs DNN (Streamlit)")
st.write(
    "Entrena una red neuronal para clasificar prendas (Fashion-MNIST). "
    "Puedes elegir arquitectura (MLP/DNN), activación, capas y ver el desempeño."
)

with st.sidebar:
    st.header("Datos")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)
    normalize = st.checkbox("Normalizar X a [0,1]", value=True)

    st.header("Modelo")
    model_kind = st.selectbox("Tipo de red", ["MLP", "DNN"])

    activation = st.selectbox("Función de activación", ["relu", "tanh", "logistic"])
    dropout = st.slider("Dropout", 0.0, 0.7, 0.2, 0.05)

    st.caption("Capas ocultas: neuronas separadas por coma. Ej: 256,128,64")
    if model_kind == "MLP":
        default_layers = "128,64"
    else:
        default_layers = "512,256,128,64"
    hidden_layers_text = st.text_input("Hidden layers", value=default_layers)

    st.header("Entrenamiento")
    epochs = st.slider("Epochs", 1, 50, 10, 1)
    batch_size = st.selectbox("Batch size", [32, 64, 128, 256], index=1)
    learning_rate = st.selectbox(
        "Learning rate",
        [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        index=2,
        format_func=lambda x: f"{x:g}",
    )
    early_stop = st.checkbox("Early stopping", value=True)
    patience = st.slider("Patience", 2, 10, 3, 1)

    train_btn = st.button("Entrenar")

# Parse layers
try:
    hidden_layers = tuple(int(x.strip()) for x in hidden_layers_text.split(",") if x.strip())
    if len(hidden_layers) == 0:
        raise ValueError("Debes definir al menos una capa oculta.")
    if any(u <= 0 for u in hidden_layers):
        raise ValueError("Las neuronas por capa deben ser enteros positivos.")
except Exception as e:
    st.error(f"Capas inválidas: {e}")
    st.stop()

# Load data
X_train, X_test, y_train, y_test = load_fashion_mnist(
    test_size=test_size,
    random_state=random_state,
    normalize=normalize,
)

# Preview
st.subheader("Vista previa de datos")
cols = st.columns(6)
for i in range(6):
    idx = np.random.randint(0, X_train.shape[0])
    img = X_train[idx].reshape(28, 28)
    cols[i].image(img, caption=CLASS_NAMES[int(y_train[idx])], clamp=True)

# Train
if train_btn:
    st.subheader("Entrenamiento y evaluación")

    with st.spinner("Construyendo y entrenando el modelo..."):
        model = build_network(
            input_dim=X_train.shape[1],
            num_classes=len(CLASS_NAMES),
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
            learning_rate=learning_rate,
        )

        history, y_pred, eval_metrics = train_and_predict(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=epochs,
            batch_size=batch_size,
            early_stop=early_stop,
            patience=patience,
        )

    st.markdown("### Métricas principales (test)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test loss", f"{eval_metrics['test_loss']:.4f}")
    c2.metric("Test accuracy", f"{eval_metrics['test_accuracy']:.4f}")
    c3.metric("Samples test", f"{len(y_test)}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Curvas (loss / accuracy)")
        fig = plt.figure(figsize=(9, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        ax1.plot(history["loss"], label="train")
        ax1.plot(history["val_loss"], label="val")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(history["accuracy"], label="train")
        ax2.plot(history["val_accuracy"], label="val")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        fig.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("### Matriz de confusión (test)")
        fig_cm = plot_confusion_matrix(y_test, y_pred, CLASS_NAMES)
        st.pyplot(fig_cm)

    st.markdown("### Classification report (test)")
    rpt_df = classification_report_df(y_test, y_pred, CLASS_NAMES)
    st.dataframe(rpt_df, use_container_width=True)

    st.markdown("### Ejemplos de predicción")
    n_show = 12
    idxs = np.random.choice(np.arange(len(X_test)), size=n_show, replace=False)
    grid_cols = 6
    rows = (n_show + grid_cols - 1) // grid_cols

    for r in range(rows):
        row_cols = st.columns(grid_cols)
        for c in range(grid_cols):
            j = r * grid_cols + c
            if j >= n_show:
                break
            idx = idxs[j]
            true_label = CLASS_NAMES[int(y_test[idx])]
            pred_label = CLASS_NAMES[int(y_pred[idx])]
            img = X_test[idx].reshape(28, 28)

            cap = f"T: {true_label}\nP: {pred_label}"
            row_cols[c].image(img, caption=cap, clamp=True)
