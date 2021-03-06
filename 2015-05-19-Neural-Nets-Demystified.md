---
layout: slide
title: Background Transitions
data:
  transition: linear
  background: '#d33682'
  background-transition: slide
---

% Neural Nets Demystified
% Hobson Lane
% May 19, 2015

# Neural Nets Demystified

1. Demystify
2. Dig Deeper

<div class="notes">

First I'll convince you that neural nets are easy to use...I'll get you going with a simple example predict Portland weather.
Then I'll show you how powerful they are and show you how to play around at the edge of what they can do.

* Thoughts about the upcoming [PDX Data Science Meetup](http://www.meetup.com/Portland-Data-Science-Group/events/222322211/)
* ["Neural Nets Demystified."](/Data-Science-Meetup--Neural-Nets-Demystified/)

---

# Classification

The most basic ML task is classification

In NN lingo, this is called "association"

So lets predict "rain" (1) "no rain" (0) for PDX tomorrow

---

# Supervised Learning

We have historical "examples" of rain and shine

[Weather Underground](http://wunderground.org)

Since we know the classification (training set)...

Supervised classification (association)

---

# Rain, Shine, Partly-Cloudy ?

Wunderground lists several possible "conditions" or classes

If we wanted to predict them all 

We would just make a binary classifier for each one

All classification problems can be reduced a binary classification

---

# [*Perceptron*](https://en.wikipedia.org/wiki/Perceptron)

Sounds mysterious, like a "flux capacitor" or something...

It's just a multiply and threshold check:

{% highlight python %}
if (weights * inputs) > 0:
    output = 1
else:
    output = 0
{% endhighlight %}

---

# Perceptron

(Diagram of a perceptron)

---

# Need something a little better

Works fine for "using" (*[activating](https://en.wikipedia.org/wiki/Activation_function)*) your NN

But for learning (*[backpropagation](https://en.wikipedia.org/wiki/Backpropagation)*) you need it to be predictable...

* doesn't change direction on you: *[monotonic](https://en.wikipedia.org/wiki/Monotonic_function)*
* doesn't jump around: *[smooth](https://en.wikipedia.org/wiki/Smoothness)*

---

# [*Sigmoid*](https://en.wikipedia.org/wiki/Perceptron)

Again, sounds mysterious... like a transcendental function

It is a transcendental function, but the word just means

Curved, smooth like the letter "C"

---

# What Greek letter do you think of when I say "Sigma"?

## "Σ"

What Roman (English) character?

* "E"?
* "S"?
* "C"?

---

# [Sigma](https://en.wikipedia.org/wiki/Sigma)

You didn't know this was a Latin/Greek class, did you...

Σ (uppercase)
σ (lowercase)
ς (last letter in word)
c (alternatively)

Most English speakers think of an "S" when they hear "Sigma" you think of an S.
So the meaning has evolved to mean S-shaped.

---

That's what we want, something smooth, shaped like an "S"

The trainer (*(backpropagator)[https://en.wikipedia.org/wiki/Backpropagation]*) can predict the change in `weights` required
Wants to nudge the `output` closer to the `target`

`target`: known classification for training examples
`output`: predicted classification your network spits out

---

# But just a nudge.

Don't get greedy and push all the way to the answer
Because your linear sloper predictions are wrong
And there may be nonlinear interactions between the weights (multiply layers)

So set the learning rate (\alpha) to somthething less than 1
the portion of the predicted nudge you want to "dial back" to

---

# Example: Predict Rain in Portland

* PyBrain
* pug-ann (helper functions TBD PyBrain2)

---

Get historical weather for Portland then ...

1. Backpropagate: train a perceptron
2. Activate: predict the weather for tomorrow!

---

NN Advantages

* Easy
    * No math!
    * No tuning!
    * Just plug and chug.
* General
    * One model can apply to many problems
* Advanced
    * They often beat all other "tuned" approaches

---

Disadvantage #1: Slow training

* 24+ hr for complex Kaggle example on laptop
* 90x30x20x10 model degrees freedom
    + 90 input dimensions (regressors)
    + 30 nodes for *hidden layer* 1
    + 20 nodes for *hidden layer* 2
    + 10 output dimensions (predicted values)

---

Disadvantage #2: They don't scale (unparallelizable)

* Fully-connected NNs can't be *easily* hyper-parallelized (GPU)
    * Large matrix multiplications
    * Layers depend on all elements of previous layers

---

Scaling Workaround

At Kaggle workshop we discussed paralleling linear algebra

  * Split matrices up and work on "tiles"
  * Theano, [Keras]() for python
  * [PLASMA](http://icl.cs.utk.edu/news_pub/submissions/plasma-scidac09.pdf) for BLAS

---

Scaling Workaround Limitations

But tiles must be shared/consolidated and theirs redundancy

  * Data flow: Main -> CPU -> GPU -> GPU cache (and back)
  * Data com (RAM xfer) is limiting
  * Data RAM size (at each stage) is limiting 
  * [Each GPU is equivalent to 16 core node](http://icl.cs.utk.edu/news_pub/submissions/plasma-scidac09.pdf)

---

Disadvantage #3: They overfit

* Too manu nodes = overfitting

---

What is the big O?

* Degrees of freedom grow with number of nodes & layers
* Each layer's nodes connected to each previous layer's
* That a lot of wasted "freedom"

# O(N^2)

---

Not so fast, big O...

{% highlight python %}
>>> np.prod([30, 20, 10])
6000
>>> np.sum([30, 20, 10])**2
3600
{% endhighlight %}


---

Rule of thumb 

NOT `N**2` 

But `M * N**2`

N: number of nodes
M: number of layers

---

`assert(M * N**2 < len(training_set) / 10.)`

I'm serious... put this into your code.
I wasted a lot of time training models for Kaggle that overfitted.

---

You do need to know math!

* To imprint your net with the structure (math) of the problem
    * Feature analysis or transformation (conventional ML)
    * Choosing the activation function and segmenting your NN
* Prune and evolve your NN

---

This is a virtuous cycle!

* More structure (no longer fully connected) 
    * Each independent path (segment) is parallelizable!
* Automatic tuning, pruning, evolving is all parallelizable!
    * Just train each NN separately
    * Check back in with Prefrontal to "compete"

---

Structure you can play with (textbook)

+ limit connections 

jargon: *receptive fields*

+ limit weights 

jargon: *weight sharing*

All the rage: *convolutional networks*

---

Unconventional structure to play with

New ideas, no jargon yet, just crackpot names

* limit weight ranges (e.g. -1 to 1, 0 to 1, etc)
* weight "snap to grid" (snap learning)

---

Joke: "What's the difference between a scientist and a crackpot?"

---

Ans: "P-value"

* High-**P**robability null hypothesis
* Not **P**ublished
* Not **P**eer-reviewed
* No **P**yPi **p**ackage

I'm a crackpot!

---

Resources

* [keras.io](http://keras.io/): Scalable Python NNs
* [Neural Network Design](http://hagan.okstate.edu/NNDesign.pdf): Free NN Textbook!
* [pug-ann](https://github.com/hobson/pug-ann): Helpers for PyBrain and Wunderground
* [PyBrain2](https://github.com/pybrain2/pybrain2): We're working on it

---

Code highlighting test

{% highlight javascript %}
function linkify( selector ) {
  if( supports3DTransforms ) {
 
    var nodes = document.querySelectorAll( selector );
 
    for( var i = 0, len = nodes.length; i &lt; len; i++ ) {
      var node = nodes[i];
 
      if( !node.className ) {
        node.className += ' roll';
      }
    }
  }
}
{% endhighlight %}

---