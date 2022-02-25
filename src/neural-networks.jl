##  Copyright 2022 Myrl Marmarelis
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.

using Flux, Zygote

make_glorot_normal(T) = (a...) -> convert.(T, Flux.glorot_normal(a...))

# is there any value in specializing the type with one final parameter
# that indicates whatever complicated structure that `model` takes on?
struct NeuralPredictor{ In, Out, T } <: Predictor{In, Out, T}
  model
end

function NeuralPredictor(n_inputs::Int, n_outputs::Int; n_layers::Int,
    n_inner_dims::Int, dropout::T) where T
  @assert n_layers >= 1
  model = Chain(
    # can the same model be used for treating all inputs,
    # or should different ones be trained for each case?
    # at least in the case of a latent space...
    Dense(n_inputs, n_inner_dims, swish, init=make_glorot_normal(T)),
    Dropout(dropout),
    ( SkipConnection( Chain(
        Dense(n_inner_dims, n_inner_dims, swish, init=make_glorot_normal(T)),
        Dropout(dropout) ), +) # make subsequent decoder layers ResNet-like
      for _ in 2:n_layers )...,
    Dense(n_inner_dims, n_outputs, init=make_glorot_normal(T)) )
  NeuralPredictor{n_inputs, n_outputs, T}(model)
end

predict(predictor::NeuralPredictor{In,Out,T},
    data::AbstractVector) where {In,Out,T} = predictor.model(data)

get_params(predictor::NeuralPredictor) = Flux.params(predictor.model)
