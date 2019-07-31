# impl-perceptron
Perceptron implementation

Data --> abalone.csv
#### Run
```
$ python main.py --epoch_count 20 --mb_size 100 --report_seq=5
```

#### Backward induction

```
loss func -> square loss L = square(y - y`)

model param (w, b) derivative  = dL/d(w, b)
    dL/dy * dy/dw = 2(y - y`) * x
        dL/dy = d(square(y - y`))/dy = 2(y - y`) * d(y - y`)/dy
              = 2(y - y`) * 1
        dy/dw = d(wx + b)/dw = x

    dL/dy * dy/db = 2(y - y`) * 1
        dL/dy = d(square(y - y`))/dy = 2(y - y`) * d(y - y`)/dy
              = 2(y - y`) * 1
        dy/db = d(wx + b)/db = l

sgd -> (w,b) - lr/mb_size * sigma dL/d(w, b)
           w - lr/mb_size * sigma 2(y - y`) * x
           b - lr/mb_size * sigma 2(y - y`) * 1
```        