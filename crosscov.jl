include("./greenfn.jl")

#################################################################
# Cross covariances and changes in cross covariances
#################################################################

module crosscov

	using Reexport,DSP
	@reexport using Main.Greenfn_radial
	import Main.Greenfn_radial: Gfn_path_from_source_radius
	@reexport using NumericallyIntegrateArray

	@reexport using PyCall,FFTW

	@reexport using Legendre,PointsOnASphere,TwoPointFunctions,VectorFieldsOnASphere

	export Gfn_fits_files
	export Cω,Cϕω,Cω_onefreq,hω,ht,Powspec,Ct,Cω_∂ϕ₂Cω,∂ϕ₂Ct,∂ϕ₂Cω
	export δCω_uniform_rotation_firstborn_integrated_over_angle
	export δCω_uniform_rotation_rotatedwaves_linearapprox
	export δCω_uniform_rotation_rotatedwaves,δCt_uniform_rotation_rotatedwaves,δCt_uniform_rotation_rotatedwaves_linearapprox
	export time_window_indices_by_fitting_bounce_peak,time_window_bounce_filter
	export @fft_ω_to_t,@fft_t_to_ω,line_of_sight

	const r_obs_default = Rsun - 75e5
	export r_obs_default

	Gfn_path_from_source_radius(x::Point3D) = Gfn_path_from_source_radius(x.r)

	function Gfn_fits_files(path,proc_id_range)
		Dict{Int64,FITS}(procid=>FITS(joinpath(path,
					@sprintf "Gfn_proc_%03d.fits" procid),"r")
					for procid in proc_id_range)
	end

	function line_of_sight(p::Point3D,detector::Point3D=Point3D(149.6e11,π/2,0))
		point_vector = CartesianVector(p)
		Detector_vector = CartesianVector(detector)
		l = HelicityVector(Detector_vector-point_vector)
		unitvector(l)
	end

	line_of_sight(n::Point2D,detector::Point3D=Point3D(149.6e11,π/2,0)) = line_of_sight(Point3D(Rsun,n),detector)

	function Powspec(ω)
		σ = 2π*0.4e-3
		ω0 = 2π*3e-3
		exp(-(ω-ω0)^2/(2σ^2))
	end

	#######################################################################################################
	
	macro fft_t_to_ω(arr,dims=1) 
		return esc(:(rfft($arr,$dims) .* dt))
	end

	macro fft_ω_to_t(arr,dims=1) 
		return esc(:(brfft($arr,$2*(size($arr,1)-1),$dims) .* dν))
	end

	#######################################################################################################

	function Cω(x1::Point3D,x2::Point3D;kwargs...)
		
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs Nν ν_start_zeros dω

		Cω_arr = zeros(ComplexF64,Nν)

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
        modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

		function summodes(ℓ_ωind_iter_on_proc)

			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			Cω_proc = zeros(ComplexF64,ν_ind_range .+ ν_start_zeros)

			Pl_cosχ = Pl(cosχ(x1,x2),ℓmax=ℓ_arr[end])

			r₁_ind = argmin(abs.(r .- x1.r))
			r₂_ind = argmin(abs.(r .- x2.r))

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(modes_in_file,ℓ_ωind_iter_on_proc)

					mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))
					ω_ind_full = ω_ind + ν_start_zeros

					ω = dω*ω_ind_full

					G = read(Gsrc_file[1],r₁_ind,:,1,1,1,mode_index)
					α_r₁ = G[1] + im*G[2]

					if r₁_ind == r₂_ind
						α_r₂ = α_r₁
					else
						G = read(Gsrc_file[1],r₂_ind,:,1,1,1,mode_index)
						α_r₂ = G[1] + im*G[2]
					end

					Cω_proc[ω_ind_full] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α_r₁) * α_r₂ * Pl_cosχ[ℓ]
					
				end
			end

			close.(values(Gfn_fits_files_src))
			return Cω_proc
		end

		T = OffsetVector{ComplexF64,Vector{ComplexF64}}
		Cω_in_range = pmapsum(T,summodes,modes_iter)

		Cω_arr[axes(Cω_in_range)...] .= Cω_in_range

		return Cω_arr
	end

	function Cω(x1::Point3D,x2_arr::Vector{<:Point3D};kwargs...)
		
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs Nν ν_start_zeros dν

		Cω_arr = zeros(ComplexF64,length(x2_arr),Nν)

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
        modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

        function summodes(ℓ_ωind_iter_on_proc)

			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			Cω_proc = zeros(ComplexF64,length(x2_arr),ν_ind_range .+ ν_start_zeros)

			Pl_cosχ = OffsetArray{Float64}(undef,ℓ_arr[end],1:length(x2_arr))
			for (ind,x2) in enumerate(x2_arr)
				Pl_cosχ[:,ind] = Pl(cosχ(x1,x2),ℓmax=ℓ_arr[end])
			end
			Pl_cosχ = copy(transpose(Pl_cosχ))

			r₁_ind = argmin(abs.(r .- x1.r))
			r₂_ind_prev = argmin(abs.(r .- x2_arr[1].r))

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(modes_in_file,ℓ_ωind_iter_on_proc)

					mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

					@assert(!isnothing(mode_index),"mode index $mode_index for ($ℓ,$ω_ind) not found by splitting across processors")

					ω_ind_full = ω_ind + ν_start_zeros
					ω = 2π*dν*ω_ind_full

					G = read(Gsrc_file[1],r₁_ind,:,1,1,1,mode_index)
					α_r₁ = G[1] + im*G[2]

					for (x2ind,x2) in enumerate(x2_arr)
						r₂_ind = argmin(abs.(r .- x2.r))
						if r₁_ind == r₂_ind
							α_r₂ = α_r₁
						elseif x2ind==1 || r₂_ind != r₂_ind_prev
							G = read(Gsrc_file[1],r₂_ind,:,1,1,mode_index)
							α_r₂ = G[1] + im*G[2]
						end
						r₂_ind_prev = r₂_ind

						Cω_proc[x2ind,ω_ind_full] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α_r₁) * α_r₂ * Pl_cosχ[ℓ]
					end
					
				end
			end

			close.(values(Gfn_fits_files_src))
			
			return Cω_proc
		end

		T = OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
		Cω_in_range = pmapsum(T,summodes,modes_iter)

		Cω_arr[axes(Cω_in_range)...] .= Cω_in_range

		copy(transpose(Cω_arr))
	end

	function Cω(n1::Point2D,n2_arr::Vector{<:Point2D};kwargs...)
		
		r_src = get(kwargs,:r_src,r_src_default)
		r_obs = get(kwargs,:r_obs,r_obs_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs Nν ν_start_zeros dν

		Cω_arr = zeros(ComplexF64,length(n2_arr),Nν)

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
        modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

        function summodes(ℓ_ωind_iter_on_proc)

			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			Cω_proc = zeros(ComplexF64,length(n2_arr),ν_ind_range .+ ν_start_zeros)

			Pl_cosχ = OffsetArray{Float64}(undef,0:ℓ_arr[end],1:length(n2_arr))
			for (n2ind,n2) in enumerate(n2_arr)
				Pl_cosχ[:,n2ind] .= Pl(cosχ(n1,n2),ℓmax=ℓ_arr[end])
			end

			Pl_cosχ = copy(transpose(Pl_cosχ))

			r_obs_ind = argmin(abs.(r .- r_obs))

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(modes_in_file,ℓ_ωind_iter_on_proc)

					mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

					ω_ind_full = ω_ind + ν_start_zeros
					ω = 2π*dν*ω_ind_full

					G = read(Gsrc_file[1],r_obs_ind,:,1,1,1,mode_index)
					abs_α_robs² = G[1]^2 + G[2]^2

					for n2ind in 1:length(n2_arr)
						Cω_proc[n2ind,ω_ind_full] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * abs_α_robs² * Pl_cosχ[n2ind,ℓ]
					end
					
				end
			end

			close.(values(Gfn_fits_files_src))
			return Cω_proc
		end

		T = OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
		Cω_in_range = pmapsum(T,summodes,modes_iter)

		Cω_arr[axes(Cω_in_range)...] .= Cω_in_range

		copy(transpose(Cω_arr))
	end

	########################################################################################################
	# Line-of-sight projected cross-covariance
	########################################################################################################

	function Cω_los(x1::Point3D,x2::Point3D;kwargs...)
		
		r_src = get(kwargs,:r_src,r_src_default)
		r_obs = get(kwargs,:r_obs,r_obs_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs Nν ν_start_zeros dν

		Cω_arr = zeros(ComplexF64,Nν)

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
        modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

        function summodes(ℓ_ωind_iter_on_proc)

			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			Cω_proc = zeros(ComplexF64,ν_ind_range .+ ν_start_zeros)

			Pl_cosχ = Pl(cosχ(x1,x2),ℓmax=ℓ_arr[end])

			r₁_ind = argmin(abs.(r .- x1.r))
			r₂_ind = argmin(abs.(r .- x2.r))

			# covariant components
			l1 = line_of_sight(x1).components
			l2 = line_of_sight(x2).components

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(modes_in_file,ℓ_ωind_iter_on_proc)

					mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

					ω_ind_full = ω_ind + ν_start_zeros
					ω = 2π*dν*ω_ind_full

					G = read(Gsrc_file[1],r₁_ind,:,:,1,mode_index)
					G0_r₁_rsrc = G[1,1] + im*G[2,1]
					G0_r₂_rsrc = G[1,2] + im*G[2,2]

					if r₁_ind == r₂_ind
						Cω_proc[ω_ind_full] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * abs2(G0_r₁_rsrc) * Pl_cosχ[ℓ]
					else
						G = read(Gsrc_file[1],r₂_ind,:,:,1,mode_index)
						G0_r₂_rsrc = G[1,1] + im*G[2,1]
						G1_r₂_rsrc = G[1,2] + im*G[2,2]
						Cω_proc[ω_ind_full] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(G0_r₁_rsrc) * G0_r₂_rsrc * Pl_cosχ[ℓ]
					end
					
				end
			end

			close.(values(Gfn_fits_files_src))
			return Cω_proc
		end

		T = OffsetArray{ComplexF64,1,Vector{ComplexF64}}
		Cω_in_range = pmapsum(T,summodes,modes_iter)

		Cω_arr[axes(Cω_in_range)...] .= Cω_in_range

		return Cω_arr
	end

	########################################################################################################
	# Add methods
	########################################################################################################

	for fn in (:Cω,:Cω_los)
		@eval $fn(n1::Point2D,n2::Point2D;r_obs::Real=r_obs_default,kwargs...) = $fn(Point3D(r_obs,n1),Point3D(r_obs,n2);kwargs...)
		@eval $fn(Δϕ::Real;r_obs::Real=r_obs_default,kwargs...) = $fn(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ);kwargs...)
		@eval $fn(n1::Point2D,n2::Point2D,ν::Real;r_obs::Real=r_obs_default,kwargs...) = $fn(Point3D(r_obs,n1),Point3D(r_obs,n2),ν;kwargs...)
		@eval $fn(Δϕ::Real,ν::Real;r_obs::Real=r_obs_default,kwargs...) = $fn(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ),ν;kwargs...)
		
		C_single_freq = quote
			function $fn(x1::Point3D,x2::Point3D,ν::Real;r_src=r_src_default,kwargs...)
			
				Gfn_path_src = Gfn_path_from_source_radius(r_src)
				@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ν_start_zeros

				ν_test_ind = argmin(abs.(ν_arr .- ν))

				$fn(x1,x2,ν_ind_range=ν_test_ind:ν_test_ind;kwargs...)[ν_start_zeros + ν_test_ind]
			end
		end
		eval(C_single_freq)
	end

	########################################################################################################
	# Spectrum of C(ℓ,ω)
	########################################################################################################    

	function Cωℓ_spectrum(;kwargs...)
		r_src = get(kwargs,:r_src,r_src_default)
		r_obs = get(kwargs,:r_obs,r_obs_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ℓ_arr num_procs Nν_Gfn dν ν_start_zeros Nν

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
        modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

		r_obs_ind = argmin(abs.(r .- r_obs))

		function summodes(ℓ_ωind_iter_on_proc)

			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			Cℓω = zeros(ℓ_range,ν_ind_range .+ ν_start_zeros)

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_file)
					
					ℓω_index_in_file = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

					ω_ind_full = ν_start_zeros + ω_ind
					ω = 2π*dν*ω_ind_full

					G = read(Gsrc_file[1],r_obs_ind,:,1,1,1,ℓω_index_in_file)
					abs_α_robs² = G[1]^2 + G[2]^2

					# m-averaged, so divided by 2ℓ+1
					Cℓω[ℓ,ω_ind_full] = ω^2 * Powspec(ω) * 1/4π * abs_α_robs²

				end
			end

			close.(values(Gfn_fits_files_src))

			return Cℓω
		end

		Cℓω = zeros(ℓ_range,Nν)

		T = OffsetArray{Float64,2,Array{Float64,2}}
		Cℓω_in_range = pmapsum(T,summodes,modes_iter)

		Cℓω[axes(Cℓω_in_range)...] .= Cℓω_in_range

		copy(transpose(Cℓω))
	end

	function Cωℓ_spectrum(ν::Real;kwargs...)
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr

		ν_test_ind = argmin(abs.(ν_arr .- ν))

		Cωℓ_spectrum(;ν_ind_range=ν_test_ind:ν_test_ind,kwargs...)
	end

	########################################################################################################
	# Derivatives of cross-covariance
	########################################################################################################

	function ∂ϕ₂Cω(x1::Point3D,x2::Point3D;kwargs...)
		
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs ν_start_zeros Nν dω

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
        modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

        function summodes(ℓ_ωind_iter_on_proc)

			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			Cω_proc = zeros(ComplexF64,ν_ind_range .+ ν_start_zeros)

			∂ϕ₂Pl_cosχ = dPl(cosχ(x1,x2),ℓmax=ℓ_range[end]) .* ∂ϕ₂cosχ(x1,x2)

			r₁_ind = argmin(abs.(r .- x1.r))
			r₂_ind = argmin(abs.(r .- x2.r))

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_file)
					
					mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

					ω_ind_full = ν_start_zeros + ω_ind 
					ω = dω*ω_ind_full
					
					G = read(Gsrc_file[1],r₁_ind,:,1,1,1,mode_index)
					α_r₁ = G[1] + im*G[2]

					if r₁_ind == r₂_ind
						α_r₂ = α_r₁
					else
						G = read(Gsrc_file[1],r₂_ind,:,1,1,1,mode_index)
						α_r₂ = G[1] + im*G[2]
					end
					Cω_proc[ω_ind_full] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α_r₁) * α_r₂ * ∂ϕ₂Pl_cosχ[ℓ]
				end
			end

			close.(values(Gfn_fits_files_src))
			return Cω_proc
		end

		Cω_arr = zeros(ComplexF64,Nν)

		T = OffsetArray{ComplexF64,1,Vector{ComplexF64}}
		Cω_in_range = pmapsum(T,summodes,modes_iter)

		Cω_arr[axes(Cω_in_range)...] .= Cω_in_range

		return Cω_arr
	end

	function ∂ϕ₂Cω(x1::Point3D,x2_arr::Vector{<:Point3D};kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs ν_start_zeros Nν dω

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
        modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

		∂ϕ₂Pl_cosχ = [dPl(cosχ(x1,x2),ℓmax=ℓ_range[end]) .* ∂ϕ₂cosχ(x1,x2) for x2 in x2_arr]

        function summodes(ℓ_ωind_iter_on_proc)

			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			Cω_proc = zeros(ComplexF64,length(x2_arr),ν_ind_range .+ ν_start_zeros)

			r₁_ind = argmin(abs.(r .- x1.r))
			r₂_ind_prev = argmin(abs.(r .- x2_arr[1].r))

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_file)
					
					ℓω_index_in_file = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

					ω_ind_full = ω_ind + ν_start_zeros
					ω = dω*ω_ind_full
					
					G = read(Gsrc_file[1],r₁_ind,:,1,1,1,ℓω_index_in_file)
					α_r₁ = G[1] + im*G[2]

					for (x2ind,x2) in enumerate(x2_arr)
						r₂_ind = argmin(abs.(r .- x2.r))

						if x2ind==1 || r₂_ind != r₂_ind_prev
							if r₂_ind == r₁_ind
								α_r₂ = α_r₁
							else
								G = read(Gsrc_file[1],r₂_ind,:,1,1,1,ℓω_index_in_file)
								α_r₂ = G[1] + im*G[2]
							end
						end
						Cω_proc[x2ind,ω_ind_full] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α_r₁) * α_r₂ * ∂ϕ₂Pl_cosχ[x2ind][ℓ]
					end
				end
			end

			close.(values(Gfn_fits_files_src))
			return Cω_proc
		end

		Cω_arr = zeros(ComplexF64,length(x2_arr),Nν)

		T = OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
		Cω_in_range = pmapsum(T,summodes,modes_iter)

		Cω_arr[axes(Cω_in_range)...] .= Cω_in_range

		copy(transpose(Cω_arr))
	end

	function ∂ϕ₂Cω(n1::Point2D,n2_arr::Vector{<:Point2D};kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		r_obs = get(kwargs,:r_obs,r_obs_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs ν_start_zeros Nν dν dω

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
        modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

		∂ϕ₂Pl_cosχ = zeros(0:ℓ_range[end],length(n2_arr))
		for (n2ind,n2) in enumerate(n2_arr)
			∂ϕ₂Pl_cosχ[:,n2ind] .= dPl(cosχ(n1,n2),ℓmax=ℓ_range[end]) .* ∂ϕ₂cosχ(n1,n2)
		end
		∂ϕ₂Pl_cosχ = copy(transpose(∂ϕ₂Pl_cosχ))

        function summodes(ℓ_ωind_iter_on_proc)

			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			Cω_proc = zeros(ComplexF64,length(n2_arr),ν_ind_range .+ ν_start_zeros)

			r_obs_ind = argmin(abs.(r .- r_obs))

			f = 0.0

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_file)
					
					ℓω_index_in_file = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

					ω_ind_full = ν_start_zeros + ω_ind
					ω = dω*ω_ind_full
					
					G = read(Gsrc_file[1],r_obs_ind,:,1,1,1,ℓω_index_in_file)
					αℓω2 = G[1]^2 + G[2]^2

					f = ω^2 * Powspec(ω) * (2ℓ+1)/4π * αℓω2

					@. Cω_proc[:,ω_ind_full] +=  f * ∂ϕ₂Pl_cosχ[:,ℓ]
					
				end
			end

			close.(values(Gfn_fits_files_src))
			return Cω_proc
		end

		Cω_arr = zeros(ComplexF64,length(n2_arr),Nν)

		T = OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
		Cω_in_range = pmapsum(T,summodes,modes_iter)

		Cω_arr[axes(Cω_in_range)...] .= Cω_in_range

		copy(transpose(Cω_arr))
	end

	function Cω_∂ϕ₂Cω(x1::Point3D,x2::Point3D;kwargs...)
		
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs ν_start_zeros Nν dω

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
        modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

        function summodes(ℓ_ωind_iter_on_proc)

			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,
											ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			Cω_proc = zeros(ComplexF64,0:1,ν_ind_range .+ ν_start_zeros)

			Pl_dPl_cosχ = Pl_dPl(cosχ(x1,x2),ℓmax=ℓ_range[end])
			Pl_cosχ = Pl_dPl_cosχ[:,0]
			dPl_cosχ = Pl_dPl_cosχ[:,1]
			∂ϕ₂Pl_cosχ =  dPl_cosχ.* ∂ϕ₂cosχ(x1,x2)

			r₁_ind = argmin(abs.(r .- x1.r))
			r₂_ind = argmin(abs.(r .- x2.r))

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_file)
					
					mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

					ω_ind_full = ν_start_zeros + ω_ind 
					ω = dω*ω_ind_full
					
					G = read(Gsrc_file[1],r₁_ind,:,1,1,1,mode_index)
					α_r₁ = G[1] + im*G[2]

					if r₁_ind == r₂_ind
						α_r₂ = α_r₁
					else
						G = read(Gsrc_file[1],r₂_ind,:,1,1,1,mode_index)
						α_r₂ = G[1] + im*G[2]
					end
					Cω_proc[0,ω_ind_full] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α_r₁) * α_r₂ * Pl_cosχ[ℓ]
					Cω_proc[1,ω_ind_full] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * conj(α_r₁) * α_r₂ * ∂ϕ₂Pl_cosχ[ℓ]
				end
			end

			close.(values(Gfn_fits_files_src))
			return Cω_proc
		end



		Cω_arr = zeros(ComplexF64,0:1,Nν)

		T = OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
		Cω_in_range = pmapsum(T,summodes,modes_iter)

		Cω_arr[axes(Cω_in_range)...] .= Cω_in_range

		Cω_arr = copy(transpose(Cω_arr.parent))

		return Cω_arr[:,1],Cω_arr[:,2]
	end

	function Cω_∂ϕ₂Cω(n1::Point2D,n2::Point2D;kwargs...)
		
		r_src = get(kwargs,:r_src,r_src_default)
		r_obs = get(kwargs,:r_obs,r_obs_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs ν_start_zeros Nν dω

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
        modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

        function summodes(ℓ_ωind_iter_on_proc)

			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			Cω_proc = zeros(ComplexF64,0:1,ν_ind_range .+ ν_start_zeros)

			Pl_dPl_cosχ = Pl_dPl(cosχ(n1,n2),ℓmax=ℓ_range[end])
			Pl_cosχ = Pl_dPl_cosχ[:,0]
			dPl_cosχ = Pl_dPl_cosχ[:,1]
			∂ϕ₂Pl_cosχ =  dPl_cosχ.* ∂ϕ₂cosχ(n1,n2)

			r_obs_ind = argmin(abs.(r .- r_obs))

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_file)
					
					mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

					ω_ind_full = ν_start_zeros + ω_ind 
					ω = dω*ω_ind_full
					
					G = read(Gsrc_file[1],r_obs_ind,:,1,1,1,mode_index)
					abs_α_robs² = G[1]^2 + G[2]^2

					Cω_proc[0,ω_ind_full] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * abs_α_robs² * Pl_cosχ[ℓ]
					Cω_proc[1,ω_ind_full] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * abs_α_robs² * ∂ϕ₂Pl_cosχ[ℓ]
				end
			end

			close.(values(Gfn_fits_files_src))
			return Cω_proc
		end



		Cω_arr = zeros(ComplexF64,0:1,Nν)

		T = OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
		Cω_in_range = pmapsum(T,summodes,modes_iter)

		Cω_arr[axes(Cω_in_range)...] .= Cω_in_range

		Cω_arr = copy(transpose(Cω_arr.parent))

		return Cω_arr[:,1],Cω_arr[:,2]
	end

	function Cω_∂ϕ₂Cω(n1::Point2D,n2_arr::Vector{<:Point2D};kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		r_obs = get(kwargs,:r_obs,r_obs_default)        
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs ν_start_zeros Nν dω

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
        modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

		∂ϕ₂Pl_cosχ = zeros(0:ℓ_range[end],length(n2_arr))
		Pl_cosχ = zeros(0:ℓ_range[end],length(n2_arr))
		for (n2ind,n2) in enumerate(n2_arr)
			Pl_dPl_cosχ = Pl_dPl(cosχ(n1,n2),ℓmax=ℓ_range[end])
			Pl_cosχ[:,n2ind] = Pl_dPl_cosχ[:,0]
			∂ϕ₂Pl_cosχ[:,n2ind] = Pl_dPl_cosχ[:,1].* ∂ϕ₂cosχ(n1,n2)
		end

		Pl_cosχ = copy(transpose(Pl_cosχ))
		∂ϕ₂Pl_cosχ = copy(transpose(∂ϕ₂Pl_cosχ))

        function summodes(ℓ_ωind_iter_on_proc)

			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,
										ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			Cω_proc = zeros(ComplexF64,0:1,length(n2_arr),ν_ind_range .+ ν_start_zeros)

			r_obs_ind = argmin(abs.(r .- r_obs))

			f = 0.0

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_file)
					
					mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

					ω_ind_full = ν_start_zeros + ω_ind 
					ω = dω*ω_ind_full
					
					G = read(Gsrc_file[1],r_obs_ind,:,1,1,1,mode_index)
					abs_α_robs² = G[1]^2 + G[2]^2

					f = ω^2 * Powspec(ω) * (2ℓ+1)/4π * abs_α_robs²

					for n2ind in 1:length(n2_arr)
						Cω_proc[0,n2ind,ω_ind_full] += f * Pl_cosχ[n2ind,ℓ]
						Cω_proc[1,n2ind,ω_ind_full] += f * ∂ϕ₂Pl_cosχ[n2ind,ℓ]
					end
				end
			end

			close.(values(Gfn_fits_files_src))
			return Cω_proc
		end

		Cω_arr = zeros(ComplexF64,0:1,length(n2_arr),Nν)

		T = OffsetArray{ComplexF64,3,Array{ComplexF64,3}}
		Cω_in_range = pmapsum(T,summodes,modes_iter)

		Cω_arr[axes(Cω_in_range)...] .= Cω_in_range

		C = copy(transpose(Cω_arr[0,:,:])).parent
		∂ϕ₂C = copy(transpose(Cω_arr[1,:,:])).parent

		return C,∂ϕ₂C
	end
	########################################################################################################
	# Time-domain cross-covariance
	########################################################################################################

	function Ct(x1,x2;kwargs...)
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dν Nt

		C = Cω(x1,x2;kwargs...)
		@fft_ω_to_t(C)
	end

	Ct(Cω_arr::Array{ComplexF64},dν) = @fft_ω_to_t(Cω_arr)
	Ct(Cω_arr::OffsetArray{ComplexF64},dν) = @fft_ω_to_t(Cω_arr.parent)

	Ct(n1,n2_arr::Vector;kwargs...) = Ct(n1,n2;kwargs...)

	function ∂ϕ₂Ct(x1,x2;kwargs...)
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dν Nt

		C = ∂ϕ₂Cω(x1,x2;kwargs...)
		@fft_ω_to_t(C)
	end

	function ∂ϕ₂Ct(n1,n2_arr::Vector;kwargs...)
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dν Nt

		C = ∂ϕ₂Cω(n1,n2_arr;kwargs...)
		# brfft(C,Nt,1) .*dν
		@fft_ω_to_t(C)
	end

	∂ϕ₂Ct(∂ϕ₂Cω_arr::AbstractArray{ComplexF64},dν) = @fft_ω_to_t(∂ϕ₂Cω_arr)

	########################################################################################################
	# Cross-covariance at all distances on the equator, essentially the time-distance diagram
	########################################################################################################

	function CΔϕω(r₁::Real=r_obs_default,r₂::Real=r_obs_default;
		ℓ_range=nothing,r_src=Rsun-75e5,Δϕ_arr=nothing,ν_ind_range=nothing,kwargs...)

		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs Nν ν_start_zeros dω

		r₁_ind = argmin(abs.(r .- r₁))
		r₂_ind = argmin(abs.(r .- r₂))

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
        modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)
		ℓmax = maximum(ℓ_range)

		if isnothing(Δϕ_arr) 
			Δϕ_arr = LinRange(0,π,ℓmax+1)[1:end-1]
		end

		nϕ = length(Δϕ_arr)

		Pl_cosχ = zeros(0:ℓmax,nϕ)
		
		for (ϕ_ind,Δϕ) in enumerate(Δϕ_arr)
			Pl_cosχ[:,ϕ_ind] = Pl(cos(Δϕ),ℓmax=ℓmax)
		end

		Pl_cosχ = collect(transpose(Pl_cosχ))

		function summodes(ℓ_ωind_iter_on_proc)

			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,
											ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			Cϕω_arr = zeros(ComplexF64,nϕ,ν_ind_range .+ ν_start_zeros)

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_file)
					
					mode_index = get_index_in_split_array(modes_in_file,(ℓ,ω_ind))

					ω_ind_full = ν_start_zeros + ω_ind
					ω = dω*ω_ind_full

					G = read(Gsrc_file[1],r₁_ind,:,1,1,1,mode_index)
					α_r₁ = G[1] + im*G[2]

					if r₁_ind == r₂_ind
						α_r₂ = α_r₁
					else
						G = read(Gsrc_file[1],r₂_ind,:,1,1,1,mode_index)
						α_r₂ = G[1] + im*G[2]
					end

					for ϕ_ind in 1:nϕ
						Cϕω_arr[ϕ_ind,ω_ind_full] += ω^2 * Powspec(ω) * (2ℓ+1)/4π * 
													conj(α_r₁) * α_r₂ * Pl_cosχ[ϕ_ind,ℓ]
					end

				end
			end

			close.(values(Gfn_fits_files_src))
			return Cϕω_arr
		end

		T = OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
		Cϕω_arr_in_range = pmapsum(T,summodes,modes_iter)

		Cϕω_arr = zeros(ComplexF64,nϕ,Nν)
		Cϕω_arr[axes(Cϕω_arr_in_range)...] .= Cϕω_arr_in_range

		return Cϕω_arr
	end

	function CtΔϕ(r₁::Real=r_obs_default,r₂::Real=r_obs_default;τ_ind_arr=nothing,kwargs...) 

		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dν Nt

		CωΔϕ = copy(transpose(CΔϕω(r₁,r₂;kwargs...)))

		C = @fft_ω_to_t(CωΔϕ)

		if isnothing(τ_ind_arr)
			return C
		else
			return C[τ_ind_arr,:]
		end
	
		return 
	end

	function CΔϕt(r₁::Real=r_obs_default,r₂::Real=r_obs_default;τ_ind_arr=nothing,kwargs...) 
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src) 
		@load joinpath(Gfn_path_src,"parameters.jld2") dν Nt

		CωΔϕ = copy(transpose(CΔϕω(r₁,r₂;kwargs...)))

		if isnothing(τ_ind_arr)
			τ_ind_arr = 1:Nt
		end
	
		return copy(transpose(@fft_ω_to_t(CωΔϕ)))[:,τ_ind_arr]
	end

	Cmω(r₁::Real=r_obs_default,r₂::Real=r_obs_default;kwargs...) = fft(CΔϕω(r₁,r₂;kwargs...),1)

	########################################################################################################
	# Cross-covariance in a rotating frame
	########################################################################################################

	function Cτ_rotating(x1::Point3D,x2::Point3D;
		Ω_rot = 20e2/Rsun,τ_ind_arr = nothing,kwargs...)
		
		# Return C(Δϕ,ω) = RFFT(C(Δϕ,τ)) = RFFT(IRFFT(C0(Δϕ - Ωτ,ω))(τ))

		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") Nt dt

		if isnothing(τ_ind_arr)
			τ_ind_arr = 1:div(Nt,2)
		end

		Nτ = length(τ_ind_arr)

		Cτ_arr = OffsetArray{Float64}(undef,τ_ind_arr)
		
		for τ_ind in τ_ind_arr
			τ = (τ_ind-1) * dt
			x2′ = Point3D(x2.r,x2.θ,x2.ϕ-Ω_rot*τ)
			Cτ_arr[τ_ind] = Ct(x1,x2′;kwargs...)[τ_ind]
		end

		return Cτ_arr
	end

	function Cτ_rotating(x1,x2_arr::Vector;
		Ω_rot = 20e2/Rsun,τ_ind_arr = nothing,kwargs...)
		
		# Return C(Δϕ,ω) = RFFT(C(Δϕ,τ)) = RFFT(IRFFT(C0(Δϕ - Ωτ,ω))(τ))

		r_src = get(kwargs,:r_src,r_src_default)
		
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") Nt dt

		if isnothing(τ_ind_arr)
			τ_ind_arr = [1:div(Nt,2) for x2 in x2_arr]
		end

		τ_ind_max_span = minimum(minimum.(τ_ind_arr)):maximum(maximum.(τ_ind_arr))

		Cτ_arr = [OffsetArray{Float64}(undef,τ_inds) for τ_inds in τ_ind_arr]
		
		for τ_ind in τ_ind_max_span
			τ = (τ_ind-1) * dt
			if T==Point3D
				x2′_arr = [T(x2.r,x2.θ,x2.ϕ-Ω_rot*τ) for (rank,x2) in enumerate(x2_arr)
							 if τ_ind in τ_ind_arr[rank]]
			elseif T==Point2D
				x2′_arr = [T(x2.θ,x2.ϕ-Ω_rot*τ) for (rank,x2) in enumerate(x2_arr)
							 if τ_ind in τ_ind_arr[rank]]
			end

			x2′inds_arr = [rank for (rank,x2) in enumerate(x2_arr) if τ_ind in τ_ind_arr[rank]]
			Ct_x2_arr = Ct(x1,x2′_arr;kwargs...)[τ_ind,:]
			
			for (Ct_x2,x2ind) in zip(Ct_x2_arr,x2′inds_arr)
				Cτ_arr[x2ind][τ_ind] = Ct_x2
			end
		end

		return Cτ_arr
	end

	#######################################################################################################################################

	function δCω_uniform_rotation_firstborn_integrated_over_angle(x1::Point3D,x2::Point3D,ν::Real;
		kwargs...)
		
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)

		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr

		ν_test_ind = argmin(abs.(ν_arr .- ν))

		δCω_uniform_rotation_firstborn_integrated_over_angle(x1,x2;
							ν_ind_range=ν_test_ind:ν_test_ind,kwargs...)
	end

	function δCω_uniform_rotation_firstborn_integrated_over_angle(x1::Point3D,x2::Point3D;
		Ω_rot = 20e2/Rsun,kwargs...)
		
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_x1 = Gfn_path_from_source_radius(x1)
		Gfn_path_x2 = Gfn_path_from_source_radius(x2)

		r₁_ind = argmin(abs.(r .- x1.r))
		r₂_ind = argmin(abs.(r .- x2.r))

		@load joinpath(Gfn_path_src,"parameters.jld2") ℓ_arr num_procs Nν_Gfn dω Nν ν_start_zeros

		num_procs_obs1 = load(joinpath(Gfn_path_x1,"parameters.jld2"),"num_procs")
		num_procs_obs2 = load(joinpath(Gfn_path_x2,"parameters.jld2"),"num_procs")

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
        modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

		ℓmax = maximum(ℓ_arr)

		∂ϕ₂Pl_cosχ = dPl(cosχ(x1,x2),ℓmax=ℓmax) .*∂ϕ₂cosχ(x1,x2)

		δC = zeros(ComplexF64,Nν)

		function summodes(ℓ_ωind_iter_on_proc)
			
			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			proc_id_range_Gobs1 = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs_obs1)
			proc_id_range_Gobs2 = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs_obs2)

			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)
			Gfn_fits_files_obs1 = Gfn_fits_files(Gfn_path_x1,proc_id_range_Gobs1)
			Gfn_fits_files_obs2 = Gfn_fits_files(Gfn_path_x2,proc_id_range_Gobs2)

			δC_r = zeros(ComplexF64,nr,ν_ind_range .+ ν_start_zeros)

			δG_r₁_rsrc = zeros(ComplexF64,nr)
			δG_r₂_rsrc = zeros(ComplexF64,nr)
			Gsrc = zeros(ComplexF64,nr,0:1)
			Gobs1 = zeros(ComplexF64,nr,0:1)
			Gobs2 = zeros(ComplexF64,nr,0:1)

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in Gsrc file
				modes_in_Gsrc_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_Gsrc_file)
					
					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_Gsrc_file,(ℓ,ω_ind))

					ω_ind_full = ω_ind + ν_start_zeros
					ω = dω*ω_ind_full
					
					G = read(Gsrc_file[1],:,:,1:2,1,1,ℓω_index_Gsrc_file)
					@. Gsrc[:,0] = G[:,1,1] + im*G[:,2,1]
					@. Gsrc[:,1] = G[:,1,2] + im*G[:,2,2]
					G_r₁_rsrc = Gsrc[r₁_ind,0]
					G_r₂_rsrc = Gsrc[r₂_ind,0]

					proc_id_mode_Gobs1,ℓω_index_Gobs1_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs1)
					Gobs1_file = Gfn_fits_files_obs1[proc_id_mode_Gobs1]

					G = read(Gobs1_file[1],:,:,1:2,1,1,ℓω_index_Gobs1_file)
					@. Gobs1[:,0] = G[:,1,1] + im*G[:,2,1]
					@. Gobs1[:,1] = G[:,1,2] + im*G[:,2,2]

					@. δG_r₁_rsrc = Gsrc[:,0] * Gobs1[:,0] - Gsrc[:,0] * Gobs1[:,1]/Ω(ℓ,0) -
									Gsrc[:,1]/Ω(ℓ,0) * Gobs1[:,0] + (ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs1[:,1]/Ω(ℓ,0)

					if r₁_ind != r₂_ind

						proc_id_mode_Gobs2,ℓω_index_Gobs2_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs2)
						Gobs2_file = Gfn_fits_files_obs2[proc_id_mode_Gobs2]

						G = read(Gobs2_file[1],:,:,1:2,1,1,ℓω_index_Gobs2_file)
						@. Gobs2[:,0] = G[:,1,1] + im*G[:,2,1]
						@. Gobs2[:,1] = G[:,1,2] + im*G[:,2,2]

						@. δG_r₂_rsrc = Gsrc[:,0] * Gobs2[:,0] - Gsrc[:,0] * Gobs2[:,1]/Ω(ℓ,0) -
									Gsrc[:,1]/Ω(ℓ,0) * Gobs2[:,0] + (ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs2[:,1]/Ω(ℓ,0)

					else
						@. δG_r₂_rsrc = δG_r₁_rsrc
					end

					@. δC_r[:,ω_ind_full] += ω^3*Powspec(ω)* (2ℓ+1)/4π * ∂ϕ₂Pl_cosχ[ℓ] * (conj(G_r₁_rsrc) * δG_r₂_rsrc + conj(δG_r₁_rsrc) * G_r₂_rsrc) 
				end
			end

			close.(values(Gfn_fits_files_src))
			close.(values(Gfn_fits_files_obs1))
			close.(values(Gfn_fits_files_obs2))

			δCω = simps((@. r^2 * ρ * δC_r),r)
			return δCω
		end

		T = OffsetVector{ComplexF64,Vector{ComplexF64}}
		δC_in_range = pmapsum(T,summodes,modes_iter)

		δC[axes(δC_in_range)...] .= δC_in_range

		return @. -2im*Ω_rot*δC
	end

	########################################################################################################################################

	function δCω_uniform_rotation_rotatedwaves_linearapprox(x1::Point3D,x2::Point3D,ν::Real;
		Ω_rot= 20e2/Rsun,kwargs...)
		
		# We compute δC(x1,x2) = -iΩ ∑ℓ (2ℓ+1)/4π ∂ω (ω^2 P(ω) αℓω*(r₂)αℓω(r₁))∂Δϕ Pl(cos(Δϕ))
		
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)

		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr dω ν_start_zeros

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)

		ν_test_ind = argmin(abs.(ν_arr .- ν))
		ν_on_grid = ν_arr[ν_test_ind]

		ν_ind_range = max(ν_test_ind-7,1):(ν_test_ind+min(7,ν_test_ind-1))
		ν_match_index = ν_test_ind - ν_ind_range[1] + 1

		∂ϕC = ∂ϕ₂Cω(x1,x2;ν_ind_range=ν_ind_range,kwargs...)[ν_ind_range .+ ν_start_zeros]

		∂ω∂ϕC = D(length(∂ϕC))*∂ϕC ./ dω

		return -im*Ω_rot*∂ω∂ϕC[ν_match_index]
	end

	function δCω_uniform_rotation_rotatedwaves_linearapprox(x1,x2;
		Ω_rot= 20e2/Rsun,kwargs...)

		# We compute δC(x1,x2) = -iΩ ∑ℓ (2ℓ+1)/4π ∂ω (ω^2 P(ω) αℓω*(r₂)αℓω(r₁))∂Δϕ Pl(cos(Δϕ))
		
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dω Nν

		∂ϕC = ∂ϕ₂Cω(x1,x2;kwargs...)

		∂ω∂ϕC = D(length(∂ϕC))*∂ϕC ./ dω

		return @. -im*Ω_rot*∂ω∂ϕC
	end

	#######################################################################################################################

	function δCt_uniform_rotation_rotatedwaves(x1::Point3D,x2::Point3D;
		Ω_rot = 20e2/Rsun,τ_ind_arr = nothing,kwargs...)
		C′_t = Cτ_rotating(x1,x2;Ω_rot=Ω_rot,τ_ind_arr=τ_ind_arr,kwargs...)
		τ_ind_arr = axes(C′_t,1)
		C0_t = Ct(x1,x2;kwargs...)[τ_ind_arr]
		return C′_t .- C0_t
	end

	function δCt_uniform_rotation_rotatedwaves(x1,x2_arr::Vector;
		Ω_rot = 20e2/Rsun,τ_ind_arr = nothing,kwargs...)

		C′_t = Cτ_rotating(x1,x2_arr;Ω_rot=Ω_rot,τ_ind_arr=τ_ind_arr,kwargs...)
		τ_ind_arr = [axes(Cx2,1) for Cx2 in C′_t]
		C0_t = [Ct(x1,x2_arr;kwargs...)[τ_ind,ind] for (ind,τ_ind) in enumerate(τ_ind_arr)]
		return C′_t .- C0_t
	end

	function δCt_uniform_rotation_rotatedwaves_linearapprox(x1::Point3D,x2::Point3D;
		Ω_rot = 20e2/Rsun,τ_ind_arr = nothing,kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path = Gfn_path_from_source_radius(r_src)
		@load "$Gfn_path/parameters.jld2" Nt dt
		
		t = (0:Nt-1).*dt
		
		δCt = -Ω_rot .* t .* ∂ϕ₂Ct(x1,x2;kwargs...)
		if !isnothing(τ_ind_arr)
			return δCt[τ_ind_arr]
		else
			return δCt
		end
	end 

	function δCt_uniform_rotation_rotatedwaves_linearapprox(x1::Point3D,x2_arr::Vector{<:Point3D};
		Ω_rot = 20e2/Rsun,τ_ind_arr = nothing,kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path = Gfn_path_from_source_radius(r_src)
		@load "$Gfn_path/parameters.jld2" Nt dt
		
		t = (0:Nt-1).*dt
		
		δCt = -Ω_rot .* t .* ∂ϕ₂Ct(x1,x2_arr;kwargs...)
		δCt_arr = Vector{Float64}[]
		if !isnothing(τ_ind_arr)
			for x2ind in 1:length(x2_arr)
				push!(δCt_arr,δCt[τ_ind_arr[x2ind],x2ind])
			end
		else
			δCt_arr = δCt
		end
		return δCt_arr
	end

	function δCt_uniform_rotation_rotatedwaves_linearapprox(n1::Point2D,n2_arr::Vector{<:Point2D};
		Ω_rot = 20e2/Rsun,kwargs...)
		
		C = ∂ϕ₂Ct(n1,n2_arr;kwargs...)
		
		δCt_uniform_rotation_rotatedwaves_linearapprox(C;Ω_rot=Ω_rot,kwargs...)

	end

	function δCt_uniform_rotation_rotatedwaves_linearapprox(∂ϕ₂Ct_arr::Array{Float64};
		Ω_rot = 20e2/Rsun,kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path = Gfn_path_from_source_radius(r_src)
		@load "$Gfn_path/parameters.jld2" Nt dt

		@. -Ω_rot * (0:Nt-1)*dt * ∂ϕ₂Ct_arr
	end

	########################################################################################################################

	function bounce_filter(Δϕ,n)
		nparams = 5
		coeffs = Dict()
		for i in [1,2,4]
			coeffs[i] = Dict("high"=>zeros(nparams),"low"=>zeros(nparams))
		end

		coeffs[1]["high"] = [2509.132334896018,12792.508296270391,-13946.527195127102,8153.75242742649,-1825.7567469552703]
		coeffs[1]["low"] = [40.821191938380714,11410.21390421857,-11116.305124138207,5254.244817703224,-895.0009393800744]

		coeffs[2]["high"] = [4083.6946001848364,14924.442447995087,-13880.238239469609,7562.499279468063,-1622.5318939228978]
		coeffs[2]["low"] = [2609.4406668522433,10536.81683213881,-7023.811081076518,2586.7238222832298,-348.9245124332354]

		coeffs[4]["high"] = [6523.103468645263,16081.024611219753,-7085.7174198723405,973.4990690666436,236.95568587146957]
		coeffs[4]["low"] = [5150.314633252216,15040.045600508669,-8813.047362534506,3878.5398150601663,-870.3633232120256]

		τ_low,τ_high = 0.,0.
		for (i,c) in enumerate(coeffs[n]["high"])
			τ_high += c*Δϕ^(i-1)
		end

		for (i,c) in enumerate(coeffs[n]["low"])
			τ_low += c*Δϕ^(i-1)
		end

		return τ_low,τ_high
	end

	function gaussian_fit(x,y)
		# use the fact that if y=Gaussian(x), log(y) = quadratic(x)
		# quadratic(x) = ax² + bx + c
		# Gaussian(x) = A*exp(-(x-x0)²/2σ²)
		# the parameters are σ=√(-1/2a), x0 = -b/2a, A=exp(c-b^2/4a)
		c,b,a=polyfit(x,log.(y),2).a
		A = exp(c-b^2/4a)
		x0 = -b/2a
		σ = √(-1/2a)
		return A,x0,σ
	end

	function time_window_bounce_filter(x1,x2,dt,bounce_no=1)
		time_window_bounce_filter(acos(cosχ(x1,x2)),dt,bounce_no)
	end

	function time_window_bounce_filter(Δϕ::Real,dt,bounce_no=1)
		τ_low,τ_high = bounce_filter(Δϕ,bounce_no)
		τ_low_ind = floor(Int64,τ_low/dt); 
		τ_high_ind = ceil(Int64,τ_high/dt)
		return τ_low_ind,τ_high_ind
	end

	function time_window_indices_by_fitting_bounce_peak(C_t::Array{Float64,1},x1,x2;
		dt,bounce_no=1,kwargs...)

		τ_low_ind,τ_high_ind = time_window_bounce_filter(x1,x2,dt,bounce_no)
		time_window_indices_by_fitting_bounce_peak(C_t,τ_low_ind,τ_high_ind;dt=dt,kwargs...)
	end

	function time_window_indices_by_fitting_bounce_peak(C_t::Array{Float64,1},τ_low_ind::Int64,τ_high_ind::Int64;
		dt,Nt=size(C_t,1),kwargs...)
		
		env = abs.(hilbert(C_t[1:div(Nt,2)]))
		peak_center = argmax(env[τ_low_ind:τ_high_ind]) + τ_low_ind - 1
		points_around_max = env[peak_center-2:peak_center+2]
		amp = env[peak_center]
		A,t0,σt = gaussian_fit(peak_center-2:peak_center+2, points_around_max)

		t_inds_range = floor(Int64,t0 - 2σt):ceil(Int64,t0 + 2σt)
	end

	function time_window_indices_by_fitting_bounce_peak(C_t::Array{Float64,2},x1,x2_arr::Vector;kwargs...)
		t_inds_range = Vector{UnitRange}(undef,size(C_t,2))
		for (x2ind,x2) in enumerate(x2_arr)
			t_inds_range[x2ind] = time_window_indices_by_fitting_bounce_peak(C_t[:,x2ind],
									x1,x2;kwargs...)
		end
		return t_inds_range
	end

	function time_window(a::Vector,τ_ind_arr)
		b = zeros(size(a))
		b[τ_ind_arr] .= 1.0
		return b
	end

	function time_window(a::Array{T,2},τ_ind_arr::Vector) where {T}
		b = zeros(size(a))
		for idx in axes(a,2)
			b[τ_ind_arr[idx],idx] .= 1.0
		end
		return b
	end

	function ht(Cω_x1x2::Array{ComplexF64},args...;bounce_no=1,kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_full Nt dt dν

		ω_full = 2π.*ν_full

		C_t = @fft_ω_to_t(Cω_x1x2)
		∂tCt = @fft_ω_to_t(Cω_x1x2.*im.*ω_full)

		τ_ind_arr = get(kwargs,:τ_ind_arr,nothing)
		if isnothing(τ_ind_arr)
			τ_ind_arr = time_window_indices_by_fitting_bounce_peak(C_t,args...;
						dt=dt,Nt=Nt,bounce_no=bounce_no)
		end

		f_t = time_window(∂tCt,τ_ind_arr)

		h_t =  (@. f_t * ∂tCt) ./ sum((@. f_t*∂tCt^2 * dt),dims=1)
	end

	function ht(x1,x2;kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") ν_full Nt

		Cω_x1x2 = Cω(x1,x2;kwargs...)
		
		ht(Cω_x1x2,x1,x2;kwargs...)
	end

	function hω(x1,x2;kwargs...)
		
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dt

		h_ω = @fft_t_to_ω(ht(x1,x2;kwargs...))
	end

	function hω(Cω_x1x2::Array{ComplexF64},args...;kwargs...)
		
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dt

		h_ω = @fft_t_to_ω(ht(Cω_x1x2,args...;kwargs...))
	end

	hω(::Array{Nothing},args...;kwargs...) = hω(args...;kwargs...)

	##########################################################################################################################
	# Add methods for computing cross-covariances in 2D (same obs radii) and 1D (same obs radii and on the equator)
	##########################################################################################################################

	for fn in (:∂ϕ₂Cω,:∂ϕ₂Ct,:Ct,:Cτ_rotating,
		:δCω_uniform_rotation_firstborn_integrated_over_angle,:δCω_uniform_rotation_firstborn_integrated_over_angle,
		:δCt_uniform_rotation_rotatedwaves,:δCt_uniform_rotation_rotatedwaves_linearapprox)

		@eval $fn(n1::Point2D,n2::Point2D;r_obs::Real=r_obs_default,kwargs...) = $fn(Point3D(r_obs,n1),Point3D(r_obs,n2);kwargs...)
		@eval $fn(Δϕ::Real;r_obs::Real=r_obs_default,kwargs...) = $fn(Point3D(r_obs,π/2,0),Point3D(r_obs,π/2,Δϕ);kwargs...)
	end

end