function a(x,y,z)
    return x+y+z
end

function b1(x,y,z)
    return x + y*z
end

function b2(x,y,z)
    return x*y + z
end

function b3(x,y,z)
    return x*z + y
end

function c(x,y,z)
    return 1+x
end

function f(x,y,z)
    return 1+y
end

function fem(x,y,z)
    local r = {}
    r["a"] = a(x,y,z)
    r["b1"] = b1(x,y,z)
    r["b2"] = b2(x,y,z)
    r["b3"] = b3(x,y,z)
    r["c"] = c(x,y,z)
    r["f"] = f(x,y,z)
    return r
end