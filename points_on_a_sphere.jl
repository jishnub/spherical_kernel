module PointsOnASphere

export Point3D,Point2D,SphericalPoint

struct Point3D
	r :: Real
	θ :: Real
	ϕ :: Real
	# Point3D(r,θ,ϕ) = ((0≤mod(θ,2π)≤π) && (0≤mod(ϕ,2π)<2π)) ? new(r,mod(θ,2π),mod(ϕ,2π)) : new(r,2π-mod(θ,2π),mod(ϕ+π,2π))
	# Point3D(θ,ϕ) = ((0≤mod(θ,2π)≤π) && (0≤mod(ϕ,2π)<2π)) ? new(1,mod(θ,2π),mod(ϕ,2π)) : new(1,2π-mod(θ,2π),mod(ϕ+π,2π))
end

struct Point2D
	θ :: Real
	ϕ :: Real
	# Point2D(θ,ϕ) = ((0≤mod(θ,2π)≤π) && (0≤mod(ϕ,2π)<2π)) ? new(mod(θ,2π),mod(ϕ,2π)) : new(2π-mod(θ,2π),mod(ϕ+π,2π))
end

Point3D(r::Real,n::Point2D) = Point3D(r,n.θ,n.ϕ)
Point3D(x::Point3D) = x

SphericalPoint = Union{Point2D,Point3D}

end