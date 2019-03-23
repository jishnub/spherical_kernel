module PointsOnASphere

export Point3D,Point2D,SphericalPoint

struct Point3D{R<:Real,T<:Real,P<:Real}
	r :: R
	θ :: T
	ϕ :: P
end

struct Point2D{T<:Real,P<:Real}
	θ :: T
	ϕ :: P
end

Point3D(r::Real,n::Point2D) = Point3D(r,n.θ,n.ϕ)
Point3D(x::Point3D) = x
Point3D(n::Point2D) = Point3D(1,n)
Point3D(n::Tuple{<:Real,<:Real}) = Point3D(1,n...)
Point3D(r::Real,n::Tuple{<:Real,<:Real}) = Point3D(r,n...)

Point2D(x::Point3D) = Point2D(x.θ,x.ϕ)
Point2D(x::Point2D) = x

Point2D(n) = Point2D(n...)
Point3D(x) = Point3D(x...)

SphericalPoint = Union{Point2D,Point3D}

# display
Base.show(io::IO, pt::Point3D) = print(io,"(r=$(round(pt.r,sigdigits=3)), θ=$(round(pt.θ,sigdigits=3)), ϕ = $(round(pt.ϕ,sigdigits=3)))")
Base.show(io::IO, pt::Point2D) = print(io,"(θ=$(round(pt.θ,sigdigits=3)), ϕ=$(round(pt.ϕ,sigdigits=3)))")

Base.show(io::IO, ::MIME"text/plain", pt::Point3D) = print(io,"(r=$(pt.r), θ=$(pt.θ), ϕ = $(pt.ϕ))")
Base.show(io::IO, ::MIME"text/plain", pt::Point2D) = print(io,"(θ=$(pt.θ), ϕ = $(pt.ϕ))")

# Precompile hints
precompile(Point3D,(Float64,Float64,Float64))
precompile(Point3D,(Float64,Float64,Int64))
precompile(Point3D,(Float64,Int64,Int64))
precompile(Point3D,(Int64,Float64,Int64))
precompile(Point3D,(Int64,Int64,Float64))
precompile(Point3D,(Int64,Float64,Int64))
precompile(Point3D,(Float64,Int64,Int64))
precompile(Point3D,(Int64,Int64,Int64))

precompile(Point2D,(Float64,Float64))
precompile(Point2D,(Float64,Int64))
precompile(Point2D,(Int64,Float64))
precompile(Point2D,(Int64,Int64))

end # module
