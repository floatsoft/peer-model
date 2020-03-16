
# coding: utf-8

# In[13]:


from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import os

data_root = "../training-data/"
data_lang = "javascript"
data = {
  'train': { 'file': data_root + 'train/' + data_lang + '/0.train.ctf', 'location': 0 },
  'test': { 'file': data_root +'test/' + data_lang + '/0.test.ctf', 'location': 0 },
  'query': { 'file': data_root + 'utils/' + 'query.wl', 'location': 1 },
  'slots': { 'file': data_root + 'utils/' + 'slots.wl', 'location': 1 },
  'intent': { 'file': data_root + 'utils/' + 'intent.wl', 'location': 1 },
  'peer_words': { 'file': data_root + 'utils/querywords/' + 'peer_words.wl', 'location': 1 },    
}
models = {
    'slots_model': None,
    'intent_model': None,    
}


# In[14]:


import math
import numpy as np

import cntk as C


# In[15]:


# setting seed
np.random.seed(0)
C.cntk_py.set_fixed_random_seed(1)
C.cntk_py.force_deterministic_algorithms()

query_wl = [line.rstrip('\n') for line in open(data['query']['file'])]
slots_wl = [line.rstrip('\n') for line in open(data['slots']['file'])]
intent_wl = [line.rstrip('\n') for line in open(data['intent']['file'])]
peer_words_wl = [line.rstrip('\n') for line in open(data['peer_words']['file'])]

# number of words in vocab, slot labels, and intent labels
vocab_size = len(query_wl) ; num_labels = len(slots_wl) ; num_intents = len(intent_wl)    

# model dimensions
input_dim  = vocab_size
label_dim  = num_labels
emb_dim    = 150
hidden_dim = 300

# Create the containers for input feature (x) and the label (y)
x = C.sequence.input_variable(vocab_size)
y = C.sequence.input_variable(num_labels)


def BiRecurrence(fwd, bwd):
    F = C.layers.Recurrence(fwd)
    G = C.layers.Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = C.splice(F(x), G(x))
    return apply_x

def create_model():
    with C.layers.default_options(initial_state=0.1):
        return C.layers.Sequential([
            C.layers.Embedding(emb_dim, name='embed'),
            BiRecurrence(C.layers.LSTM(hidden_dim//2),
                                  C.layers.LSTM(hidden_dim//2)),
            C.layers.Dense(num_labels, name='classify')
        ])


# In[16]:


def create_reader(path, is_training):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
         query         = C.io.StreamDef(field='S0', shape=vocab_size,  is_sparse=True),
         intent        = C.io.StreamDef(field='S1', shape=num_intents, is_sparse=True),  
         slot_labels   = C.io.StreamDef(field='S2', shape=num_labels,  is_sparse=True)
     )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)


# In[17]:


reader = create_reader(data['train']['file'], is_training=True)
reader.streams.keys()


# In[18]:


def create_criterion_function_preferred(model, labels):
    ce   = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error      (model, labels)
    return ce, errs # (model, labels) -> (loss, error metric)


# In[19]:


def train_test(train_reader, test_reader, model_func, stream_name="intent", max_epochs=10):
    
    # Instantiate the model function; x is the input (feature) variable 
    model = model_func(x)
    
    # Instantiate the loss and error function
    global y
    loss, label_error = create_criterion_function_preferred(model, y)

    # training config
    epoch_size = 18000        # 18000 samples is half the dataset size 
    minibatch_size = 70
    
    # LR schedule over epochs 
    # In CNTK, an epoch is how often we get out of the minibatch loop to
    # do other stuff (e.g. checkpointing, adjust learning rate, etc.)
    # (we don't run this many epochs, but if we did, these are good values)
    lr_per_sample = [0.003]*4+[0.0015]*24+[0.0003]
    lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
    lr_schedule = C.learning_rate_schedule(lr_per_minibatch, C.UnitType.minibatch, epoch_size)
    
    # Momentum schedule
    momentum_as_time_constant = C.momentum_as_time_constant_schedule(700)
    
    # We use a the Adam optimizer which is known to work well on this dataset
    # Feel free to try other optimizers from 
    # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner
    learner = C.adam(parameters=model.parameters,
                     lr=lr_schedule,
                     momentum=momentum_as_time_constant,
                     gradient_clipping_threshold_per_sample=15, 
                     gradient_clipping_with_truncation=True)

    # Setup the progress updater
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    
    # Uncomment below for more detailed logging
    #progress_printer = ProgressPrinter(freq=100, first=10, tag='Training', num_epochs=max_epochs) 

    # Instantiate the trainer
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)

    # process minibatches and perform model training
    C.logging.log_number_of_parameters(model)

    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = train_reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
                x: train_reader.streams.query,
                y: train_reader.streams[stream_name]
            })
            trainer.train_minibatch(data)               # update model with it
            t += data[y].num_samples                    # samples so far
        trainer.summarize_training_progress()
    
    while True:
        minibatch_size = 500
        data = test_reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
            x: test_reader.streams.query,
            y: test_reader.streams[stream_name]
        })
        if not data:                                 # until we hit the end
            break
        trainer.test_minibatch(data)
    
    trainer.summarize_test_progress()


# In[20]:


def do_train_test(stream_name, model_name):
    models[model_name] = create_model()
    train_reader = create_reader(data['train']['file'], is_training=True)
    test_reader = create_reader(data['test']['file'], is_training=False)
    train_test(train_reader, test_reader, models[model_name], stream_name)


# In[21]:


do_train_test("slot_labels", "slots_model")


# In[30]:


# load dictionaries
query_dict = {query_wl[i]:i for i in range(len(query_wl))}
slots_dict = {slots_wl[i]:i for i in range(len(slots_wl))}
intent_dict = {intent_wl[i]:i for i in range(len(intent_wl))}
undefined_word = peer_words_wl[0]

# let's run a sequence through
#seq = "BOS let i = 10000 ; EOS"
seq = "BOS if ( 0n ) { console . log ( ' Hello from the if ! ' ) } else { console . log ( ' Hello from the else ! ' ) } EOS"
#seq = "BOS Array . isArray ( { pin , id , i , key , name } ) ; EOS"
#seq = "BOS const obj = { a : 1 , b : 20000 , c : 1 , d : 2 } ; EOS"
#seq = "BOS const arr = [ a , 1 , b , 20000 , c , 1 , d , 2 ] ; EOS"
#seq = "BOS const aReallyBigNumber ; EOS"
#seq = "BOS var test = ' cool ' ; EOS"
w = [query_dict[w] if w in query_dict else query_dict[undefined_word]
     for w in seq.split()] # convert to word indices
print(seq, w)
onehot = np.zeros([len(w),len(query_dict)], np.float32)
for t in range(len(w)):
    onehot[t,w[t]] = 1

#x = C.sequence.input_variable(vocab_size)
pred = models['slots_model'](x).eval({x:[onehot]})[0]
print(pred.shape)
best = np.argmax(pred,axis=1)
print(best)
list(zip(seq.split(),[slots_wl[s] for s in best]))


# In[23]:


# x = C.sequence.input_variable(vocab_size)
# y = C.input_variable(num_intents)

# def create_model():
#     with C.layers.default_options(initial_state=0.1):
#         return C.layers.Sequential([
#             C.layers.Embedding(emb_dim, name='embed'),
#             C.layers.Stabilizer(),
#             C.layers.Fold(C.layers.LSTM(hidden_dim), go_backwards=False),
#             C.layers.Dense(num_intents, name='classify')
#         ])


# In[24]:


# do_train_test("intent", "intent_model")


# In[ ]:


# let's run a sequence through
#seq = "BOS const aReallyBigNumber = ( 100 - 1 ) * 100 ; EOS"
# seq = "BOS var tmp = arr [ i ] ; EOS"
# w = [query_dict[w] if w in query_dict else query_dict[undefined_word]
#      for w in seq.split()] # convert to word indices
# onehot = np.zeros([len(w),len(query_dict)], np.float32)
# for t in range(len(w)):
#     onehot[t,w[t]] = 1

# pred = models["intent_model"](x).eval({x:[onehot]})[0]
# best = np.argmax(pred)
# print(seq, ":", intent_wl[best])


# In[ ]:




