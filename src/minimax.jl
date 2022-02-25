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

no_dynamics(T) = (_...) -> (T(-Inf), T(Inf))

lipschitz_condition(slope) =
  ((prev_point, prev_weight), point, (succ_point, succ_weight)) -> begin
    prev_range = slope * (point - prev_point)
    succ_range = slope * (succ_point - point)
    if isnan(prev_point)
      (succ_weight - succ_range, succ_weight + succ_range)
    elseif isnan(succ_point)
      (prev_weight - prev_range, prev_weight + prev_range)
    else
      upper = min(prev_weight + prev_range, succ_weight + succ_range)
      lower = max(prev_weight - prev_range, succ_weight - succ_range)
      (lower, upper)
    end
  end


# `values` need not be monotonic on `points`
# on arbitrary closures, so that it's recompiled for each specialized call
function bound_expectation(static_bounds::Function,
    points::AbstractVector{T}, values::AbstractVector{T};
    dynamic_bounds::Function=no_dynamics(T))::NTuple{2,T} where T
  ordering = sortperm(values)
  points = points[ordering] # method to ensure this is done in place?
  values = values[ordering]
  maximized = maximize_expectation(static_bounds, points, values;
    dynamic_bounds).expectation
  reverse!(points)
  reverse!(values)
  values .*= -1
  minimized = -maximize_expectation(static_bounds, points, values;
    dynamic_bounds).expectation
  (minimized, maximized)
end

# assumes `points` are sorted along `values` here
function maximize_expectation(static_bounds::Function,
    points::AbstractVector{T}, values::AbstractVector{T};
    dynamic_bounds::Function=no_dynamics(T)) where T
  n_draws = length(points)
  weights = [ static_bounds(point)[2] for point in points ]
  point_ordering = sortperm(points)
  point_neighbors = Dict( ( point_ordering[p] =>
      (get(point_ordering, p-1, 0), get(point_ordering, p+1, 0)) )
    for p in 1:n_draws )
  n_iterations = 0
  mutation = true
  while mutation
    n_iterations += 1
    mutation = false
    for index in 1:n_draws
      (this_point, this_value) = (points[index], values[index])
      # all this noise in order to enforce `points` continuity for non-monotonic `values`
      (prev_index, succ_index) = point_neighbors[index]
      previous = (prev_index > 0 ?
        (points[prev_index], weights[prev_index]) : T.((NaN, NaN)) )
      successive = (succ_index > 0 ?
        (points[succ_index], weights[succ_index]) : T.((NaN, NaN)) )
      # we call the below only on legal neighbor assignments.
      (dynamic_lo, dynamic_up) = dynamic_bounds(previous, this_point, successive)
      (static_lo, static_up) = static_bounds(this_point)
      unscaled_derivative = sum(1:n_draws) do other
        weights[other] * (this_value - values[other])
      end
      if unscaled_derivative < 0
        new_weight = max(static_lo, dynamic_lo)
        if !isapprox(new_weight, weights[index])
          mutation = true
        end
        weights[index] = new_weight
      else
        # done in this one iteration; move along
        break
      end
    end
  end
  expectation = sum(weights) \ sum(1:n_draws) do index
    weights[index] * values[index]
  end
  (; expectation, weights, n_iterations )
end
