# deslant.jl (translated from python implementation by Thomas Breuel [UniKL] )
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
using MNIST, Stage, PyCall
@pyimport scipy.ndimage.interpolation as interpolation

# Helpers
log = Log(STDERR)

# ----------------------------------------------------------------------------------------------------------------
# setup MNIST task
# ----------------------------------------------------------------------------------------------------------------
const c0 = [ i for i = 0:27, j = 0:27 ]
const c1 = [ j for i = 0:27, j = 0:27 ]

function moments(image)
  si  = sum(image)
  m0  = sum(c0 .* image)/si
  m1  = sum(c1 .* image)/si
  m00 = sum((c0 - m0) .* (c0 - m0) .* image)/si
  m11 = sum((c1 - m1) .* (c1 - m1) .* image)/si
  m01 = sum((c0 - m0) .* (c1 - m1) .* image)/si
  return [ m0, m1 ], [ m00 m01; m01 m11 ]
end

function deskew(matrix)
  output = zeros(size(matrix))
  for j = 1:size(matrix, 2)
    image   = reshape(matrix[:, j], (28, 28))
    c, v    = moments(image)
    alpha   = v[1, 2] / v[1, 1]
    affine  = [1.0 0.0; alpha 1.0]
    ocenter = [ size(image, 1)/2.0, size(image, 2)/2.0 ]
    offset  = c - (affine * ocenter)
    mat     = interpolation.affine_transform(image, affine, offset = offset, order = 1)
    for i = 1:size(output, 1)
      output[i, j] = mat[i]
    end
  end

  return output
end

@info log "reading training data"
const train_raw, train_truth = traindata()
const test_raw, test_truth   = testdata()
const classes                = Dict{Float64, Int32}()
const train                  = deskew(train_raw)
const test                   = deskew(test_raw)

f = open("mnist-deslant.jd", "w")
serialize(f, train)
serialize(f, train_truth)
serialize(f, test)
serialize(f, test_truth)
close(f)

