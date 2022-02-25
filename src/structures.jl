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

using StaticArrays, Setfield

"""
  `input` features should be scaled between zero and one,
  preferably with a mapping defined a priori (not an empirical
  CDF or standardization,) so that new `Instance`s may come
  online without having to alter the old ones.
  Note: should I store it log-transformed, then?
  Some features need not be scaled like that if they will
  never be treated.
""" # if they're already Z-scored, should I apply a normal CDF
# or an empirical CDF that guarantees the histogram to be uniform?
struct Instance{ In, Out, T<:Real }
  id :: Int
  input :: SVector{In, T}
  output :: SVector{Out, T}
end

# some differential operator on `input` must match `output`,
# the latter of which takes whatever dimensionality of the operator.
# `output` basically can be used to store intermediate computations
struct ConstraintInstance{ In, Out, T<:Real }
  input :: SVector{In, T}
  output :: SVector{Out, T}
end

wrap_treatment_index(index, In) = index < 0 ? (index + In+1) : index

# may not be the route we always follow in structuring treatment-effect models
function treat_instance(instance::Instance{In,Out,T},
    index::Int, treatment::T; perturb::Bool=false
    )::Instance{In,Out,T} where {In,Out,T}
  @assert index != 0
  index = wrap_treatment_index(index, In)
  if perturb
    @set instance.input[index] += treatment
  else
    @set instance.input[index] = treatment
  end
end

function make_propensity_instance(instance::Instance{In,Out,T},
    index::Int)::Instance{In-1,1,T} where {In,Out,T}
  index = wrap_treatment_index(index, In)
  propensity_input = SVector{In-1}(
    instance.input[i < index ? i : (i+1)] for i in 1:(In-1) )
  propensity_output = @SVector [ instance.input[index] ]
  Instance(instance.id, propensity_input, propensity_output)
end
