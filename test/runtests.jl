# runtest.jl
#
# author: Wade Shen
# swade@ll.mit.edu
# Copyright &copy; 2009 Massachusetts Institute of Technology, Lincoln Laboratory
# version 0.1
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
using ollam
using MNIST, Stage
import Base: length, start, done, next

# Helpers
log = Log(STDERR)

immutable EachCol{T}
  matrix :: Array{T, 2}
end
length(e :: EachCol)      = size(e.matrix, 2)
start(e :: EachCol)       = 1
next(e :: EachCol, state) = (e.matrix[:, state], state + 1)
done(e :: EachCol, state) = state > length(e) ? true : false

# setup MNIST task
@info log "reading training data"
train, train_truth = traindata()
test, test_truth   = testdata()
classes            = Dict{Integer, Int32}()
i                  = 1
for t in test_truth
  if !(t in keys(classes))
    classes[t] = i
    i += 1
  end
end
@info log "truth set: $classes"

# setup models, train and evaluate
@timer log "training linear SVM model" model = train_svm(EachCol(train[:, 1:60000]), train_truth[1:60000]; C = 0.01, cache_size = 250.0, eps = 0.1, shrinking = false, verbose = false)
@info log @sprintf("SVM test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)

init  = LinearModel(classes, length(train[:, 1]))
@timer log "training perceptron model" model = train_perceptron(EachCol(train), train_truth, init; iterations = 20, average = false)
@info log @sprintf("perceptron test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)

init  = LinearModel(classes, length(train[:, 1]))
@timer log "training averaged perceptron model" model = train_perceptron(EachCol(train), train_truth, init; iterations = 20, average = true)
@info log @sprintf("averaged perceptron test set error rate: %7.3f%%", test_classification(model, EachCol(test), test_truth) * 100.0)
