import tensorflow as tf
from tensorflow.keras import layers

data_root = "training-data/"
data_lang = "javascript"
data = {
    "train": {"file": data_root + "train/" + data_lang + "/0.train.ctf", "location": 0},
    "test": {"file": data_root + "test/" + data_lang + "/0.test.ctf", "location": 0},
    "query": {"file": data_root + "utils/" + "query.wl", "location": 1},
    "slots": {"file": data_root + "utils/" + "slots.wl", "location": 1},
    "intent": {"file": data_root + "utils/" + "intent.wl", "location": 1},
    "peer_words": {
        "file": data_root + "utils/querywords/" + "peer_words.wl",
        "location": 1,
    },
}

models = {
    "slots_model": None,
    "intent_model": None,
}

query_wl = [line.rstrip("\n") for line in open(data["query"]["file"])]
slots_wl = [line.rstrip("\n") for line in open(data["slots"]["file"])]
intent_wl = [line.rstrip("\n") for line in open(data["intent"]["file"])]
peer_words_wl = [line.rstrip("\n") for line in open(data["peer_words"]["file"])]

vocab_size = len(query_wl)
num_labels = len(slots_wl)
num_intents = len(intent_wl)

input_dim = vocab_size
label_dim = num_labels
emb_dim = 150
hidden_dim = 300

model = tf.keras.Sequential()

model.add(
    layers.Bidirectional(
        layers.LSTM(hidden_dim // 2, return_sequences=True), input_shape=(35, 18)
    )
)
model.add(layers.Bidirectional(layers.LSTM(hidden_dim // 2)))
model.add(layers.Dense(10))

model.summary()
