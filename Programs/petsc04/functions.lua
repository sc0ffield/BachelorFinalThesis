function compute_boundary(rank, base)
    local result = (rank + 1) * base * 0.5
    return result
end

function compute_interior(rank, base)
    local result = (rank + 1) * base * 1.5
    return result
end
