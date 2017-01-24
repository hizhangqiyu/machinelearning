# neural-nettwork-for-machine-learning
## week3

    formula: 
        w = w + (target - prediction) * x

## week6

    key words:
        full-batch
        mini-batch: better for large data set
        adaptive learning rate
        rporp

## week7

```python
def Caculate(Wxh, Whh, Why, hBias, yBias, x0, x1, x2, t0, t1, t2):
    z0 = Wxh * x0 + hBias
    h0 = Logistic(z0)
    y0 = Why * h0 + yBias
    E0 = math.pow(t0-y0, 2) / 2

    z1 = Wxh * x1 + Whh * h0 + hBias
    h1 = Logistic(z1)
    y1 = Why * h1 + yBias
    E1 = math.pow(t1-y1, 2) / 2

    z2 = Wxh * x2 + Whh * h1 + hBias
    h2 = Logistic(z2)
    y2 = Why * h2 + yBias
    E2 = math.pow(t2-y2, 2) / 2

    E = E0 + E1 + E2
```
