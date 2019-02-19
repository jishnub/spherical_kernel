module PointsOnASphere

export Point3D,Point2D,SphericalPoint

struct Point3D
	r :: Real
	θ :: Real
	ϕ :: Real
end

struct Point2D
	θ :: Real
	ϕ :: Real
end

Point3D(r::Real,n::Point2D) = Point3D(r,n.θ,n.ϕ)
Point3D(x::Point3D) = x
Point3D(n::Point2D) = Point3D(1,n)

Point2D(x::Point3D) = Point2D(x.θ,x.ϕ)
Point2D(x::Point2D) = x

SphericalPoint = Union{Point2D,Point3D}

end # module
