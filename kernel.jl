include("./crosscov.jl")

#################################################################
# Kernels and travel times
#################################################################

module kernel

	using Reexport
	@reexport using Main.crosscov
	# @pyimport scipy.integrate as integrate
	import WignerSymbols: clebschgordan
	using WignerD, FileIO
	export kernel_uniform_rotation_uplus,flow_axisymmetric_without_los,
			meridional_flow_ψ_srange

	################################################################################################################
	# Validation for uniform rotation
	################################################################################################################

	function kernel_uniform_rotation_uplus(x1::Point3D,x2::Point3D;kwargs...)
		
		r_src = get(kwargs,:r_src,r_src_default)

		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_x1 = Gfn_path_from_source_radius(x1)
		Gfn_path_x2 = Gfn_path_from_source_radius(x2)

		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs dν ν_start_zeros dω Nν_Gfn

		num_procs_obs1 = load(joinpath(Gfn_path_x1,"parameters.jld2"),"num_procs")
		num_procs_obs2 = load(joinpath(Gfn_path_x2,"parameters.jld2"),"num_procs")

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
		modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

		h_ω = get(kwargs,:hω,hω(x1,x2;r_src=r_src,kwargs...)) :: Vector{ComplexF64}
		h_ω = h_ω[ν_start_zeros .+ ν_ind_range]

		r₁_ind = argmin(abs.(r .- x1.r))
		r₂_ind = argmin(abs.(r .- x2.r))

		function summodes(ℓ_ωind_iter_on_proc)
		
			proc_id_range = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)

			∂ϕ₂Pl_cosχ = dPl(cosχ(x1,x2),ℓmax=ℓ_arr[end]).*∂ϕ₂cosχ(x1,x2)

			K = zeros(nr)

			δG_r₁_rsrc = zeros(ComplexF64,nr) # This is not actually δG, it's just a radial function dependent on f

			G = zeros(nr,2,2)
			Gsrc = zeros(ComplexF64,1:nr,0:1)
			Gobs1 = zeros(ComplexF64,1:nr,0:1)

			δG_r₂_rsrc = zeros(ComplexF64,nr)
			Gobs2 = zeros(ComplexF64,1:nr,0:1)			

			for proc_id in proc_id_range

				Gsrc_file = FITS(joinpath(Gfn_path_src,@sprintf "Gfn_proc_%03d.fits" proc_id),"r")				
				# Get a list of (ℓ,ω) in this file
				modes_in_Gsrc_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_Gsrc_file)

					# Get index of this mode in the file
					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_Gsrc_file,(ℓ,ω_ind))
					ω = dω*(ν_start_zeros + ω_ind)

					# Green function about source location
		    		G .= read(Gsrc_file[1],:,:,1:2,1,1,ℓω_index_Gsrc_file)
		    		@. Gsrc[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		    		G_r₁_rsrc = Gsrc[r₁_ind,0]
		    		G_r₂_rsrc = Gsrc[r₂_ind,0]

		    		# Green function about receiver location
		    		proc_id_mode_Gobs1,ℓω_index_Gobs1_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs1)
		    		Gobs1_file = FITS(joinpath(Gfn_path_x1,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs1),"r")

		    		G .= read(Gobs1_file[1],:,:,1:2,1,1,ℓω_index_Gobs1_file)
		    		@. Gobs1[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gobs1[:,1] = G[:,1,2] + im*G[:,2,2]

		    		@. δG_r₁_rsrc = Gsrc[:,0] * Gobs1[:,0] - Gsrc[:,0] * Gobs1[:,1]/Ω(ℓ,0) -
		    						Gsrc[:,1]/Ω(ℓ,0) * Gobs1[:,0] + (ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs1[:,1]/Ω(ℓ,0)

					close(Gobs1_file)

				    if r₁_ind != r₂_ind

				    	proc_id_mode_Gobs2,ℓω_index_Gobs2_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs2)

				    	Gobs2_file = FITS(joinpath(Gfn_path_x2,@sprintf "Gfn_proc_%03d.fits" proc_id_mode_Gobs2),"r")
			    		G .= read(Gobs2_file[1],:,:,1:2,1,1,ℓω_index_Gobs2_file)
			    		@. Gobs2[:,0] = G[:,1,1] + im*G[:,2,1]
			    		@. Gobs2[:,1] = G[:,1,2] + im*G[:,2,2]

			    		@. δG_r₂_rsrc = Gsrc[:,0] * Gobs2[:,0] - Gsrc[:,0] * Gobs2[:,1]/Ω(ℓ,0) -
		    						Gsrc[:,1]/Ω(ℓ,0) * Gobs2[:,0] + 
		    						(ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs2[:,1]/Ω(ℓ,0)

					    close(Gobs2_file)
					else
					    @. δG_r₂_rsrc = δG_r₁_rsrc
					end

					@. K +=  dω/2π * ω^3 * Powspec(ω) * (2ℓ+1)/4π * ∂ϕ₂Pl_cosχ[ℓ] * imag(h_ω[ω_ind]) * 
								(conj(δG_r₁_rsrc)*G_r₂_rsrc + conj(G_r₁_rsrc)*δG_r₂_rsrc) 

				end

				close(Gsrc_file)
			end

			@. K *= -4*√(3/4π) * r * ρ
			
			return K
		end

		pmapsum(Vector{Float64},summodes,modes_iter)
	end

	function kernel_uniform_rotation_uplus(n1::Point2D,n2::Point2D;kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		r_obs = get(kwargs,:r_obs,r_obs_default)

		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_obs = Gfn_path_from_source_radius(r_obs)

		@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr ℓ_arr num_procs dω ν_start_zeros Nν_Gfn

		num_procs_obs = load(joinpath(Gfn_path_obs,"parameters.jld2"),"num_procs")

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
		modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

		r_obs_ind = argmin(abs.(r .- r_obs))

		∂ϕ₂Pl_cosχ = dPl(cosχ(n1,n2),ℓmax=ℓ_arr[end]).*∂ϕ₂cosχ(n1,n2)
		
		h_ω = get(kwargs,:hω,nothing)
		if isnothing(h_ω)
			h_ω = hω(n1,n2;r_src=r_src,r_obs=r_obs,kwargs...)
		end
		h_ω = h_ω[ν_start_zeros .+ ν_ind_range]

		function summodes(ℓ_ωind_iter_on_proc)

			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			proc_id_range_Gobs = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs_obs)

			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)
			Gfn_fits_files_obs = Gfn_fits_files(Gfn_path_obs,proc_id_range_Gobs)

			K = zeros(nr)

			δG_robs_rsrc = zeros(ComplexF64,nr) # This is not actually δG, it's just a radial function dependent on f

			G = zeros(nr,2,2)
			Gsrc = zeros(ComplexF64,nr,0:1)
			Gobs = zeros(ComplexF64,nr,0:1)

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_Gsrc_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_Gsrc_file)

					# Get index of this mode in the file
					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_Gsrc_file,(ℓ,ω_ind))
					ω = dω*(ν_start_zeros + ω_ind)

					# Green function about source location
		    		G .= read(Gsrc_file[1],:,:,1:2,1,1,ℓω_index_Gsrc_file)
		    		@. Gsrc[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		    		G_robs_rsrc = Gsrc[r_obs_ind,0]

		    		# Green function about receiver location
		    		
		    		proc_id_mode_Gobs,ℓω_index_Gobs_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ,ω_ind),num_procs_obs)
		    		Gobs_file = Gfn_fits_files_obs[proc_id_mode_Gobs]

		    		G .= read(Gobs_file[1],:,:,1:2,1,1,ℓω_index_Gobs_file)
		    		@. Gobs[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gobs[:,1] = G[:,1,2] + im*G[:,2,2]

		    		@. δG_robs_rsrc = Gsrc[:,0] * Gobs[:,0] - Gsrc[:,0] * Gobs[:,1]/Ω(ℓ,0) -
		    						Gsrc[:,1]/Ω(ℓ,0) * Gobs[:,0] + 
		    						(ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs[:,1]/Ω(ℓ,0)


					@. K +=  dω/2π * ω^3 * Powspec(ω) * 
							(2ℓ+1)/4π * ∂ϕ₂Pl_cosχ[ℓ] * imag(h_ω[ω_ind]) * 
								2real(conj(δG_robs_rsrc)*G_robs_rsrc) 

				end
			end

			close.(values(Gfn_fits_files_src))
			close.(values(Gfn_fits_files_obs))

			@. K *= -4*√(3/4π) * r * ρ

			return K
		end

		pmapsum(Vector{Float64},summodes,modes_iter)
	end
	
	function kernel_uniform_rotation_uplus(n1::Point2D,n2_arr::Vector{<:Point2D};
		hω_arr=nothing,kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		r_obs = get(kwargs,:r_obs,r_obs_default)

		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_obs = Gfn_path_from_source_radius(r_obs)

		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs dω ν_start_zeros

		num_procs_obs = load(joinpath(Gfn_path_obs,"parameters.jld2"),"num_procs")

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
		modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)

		r_obs_ind = argmin(abs.(r .- r_obs))

		∂ϕ₂Pl_cosχ_arr = zeros(0:ℓ_arr[end],length(n2_arr))
		for (n2ind,n2) in enumerate(n2_arr)
			∂ϕ₂Pl_cosχ_arr[:,n2ind] = dPl(cosχ(n1,n2),ℓmax=ℓ_arr[end]).*∂ϕ₂cosχ(n1,n2)
		end
		∂ϕ₂Pl_cosχ_arr = copy(transpose(∂ϕ₂Pl_cosχ_arr))
		
		if  isnothing(hω_arr)
			hω_arr = zeros(ComplexF64,ν_ind_range,length(n2_arr))
			τ_ind_arr = get(kwargs,:τ_ind_arr,[nothing for i in n2_arr])
			Cω_arr=get(kwargs,:Cω_arr,Array{Nothing,2}(undef,1,length(n2_arr)))
			for (n2ind,n2) in enumerate(n2_arr)
				hω_arr[ν_ind_range,n2ind] = hω(Cω_arr[:,n2ind],n1,n2;
				τ_ind_arr=τ_ind_arr[n2ind],
				r_src=r_src,r_obs=r_obs,kwargs...)[ν_start_zeros .+ ν_ind_range]
			end
			hω_arr = copy(transpose(hω_arr))
		else
			hω_arr = copy(transpose(hω_arr))[:,ν_start_zeros .+ ν_ind_range]
		end

		function summodes(ℓ_ωind_iter_on_proc)
		
			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			proc_id_range_Gobs = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs_obs)

			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)
			Gfn_fits_files_obs = Gfn_fits_files(Gfn_path_obs,proc_id_range_Gobs)

			K = zeros(nr,length(n2_arr))

			δG_robs_rsrc = zeros(ComplexF64,nr) # This is not actually δG, it's just a radial function dependent on f

			G = zeros(nr,2,2)
			Gsrc = zeros(ComplexF64,nr,0:1)
			Gobs = zeros(ComplexF64,nr,0:1)

			fr = zeros(nr)

			∂ϕ₂Pl_cosχ = zeros(0:ℓ_arr[end])

			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_Gsrc_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_Gsrc_file)

					# Get index of this mode in the file
					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_Gsrc_file,(ℓ,ω_ind))
					ω = dω*(ν_start_zeros + ω_ind)

					# Green function about source location
		    		G .= read(Gsrc_file[1],:,:,1:2,1,1,ℓω_index_Gsrc_file)
		    		@. Gsrc[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gsrc[:,1] = G[:,1,2] + im*G[:,2,2]
		    		G_robs_rsrc = Gsrc[r_obs_ind,0]

		    		# Green function about receiver location
		    		proc_id_mode_Gobs,ℓω_index_Gobs_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,
		    												(ℓ,ω_ind),num_procs_obs)
		    		Gobs_file = Gfn_fits_files_obs[proc_id_mode_Gobs]

		    		G .= read(Gobs_file[1],:,:,1:2,1,1,ℓω_index_Gobs_file)
		    		@. Gobs[:,0] = G[:,1,1] + im*G[:,2,1]
		    		@. Gobs[:,1] = G[:,1,2] + im*G[:,2,2]

		    		@. δG_robs_rsrc = Gsrc[:,0] * Gobs[:,0] - Gsrc[:,0] * Gobs[:,1]/Ω(ℓ,0) -
		    						Gsrc[:,1]/Ω(ℓ,0) * Gobs[:,0] + 
		    						(ℓ*(ℓ+1)-1) * Gsrc[:,1]/Ω(ℓ,0) * Gobs[:,1]/Ω(ℓ,0)

		    		@. fr = dω/2π * ω^3 * Powspec(ω) * 2real(conj(δG_robs_rsrc)*G_robs_rsrc) 

					for n2ind in 1:length(n2_arr)
						@. K[:,n2ind] +=   fr * (2ℓ+1)/4π * ∂ϕ₂Pl_cosχ_arr[n2ind,ℓ] * 
											imag(hω_arr[n2ind,ω_ind])
					end
					

				end
			end

			close.(values(Gfn_fits_files_src))
			close.(values(Gfn_fits_files_obs))

			@. K *= -4*√(3/4π) * r * ρ

			return K
		end

		pmapsum(Array{Float64,2},summodes,modes_iter)
	end

	################################################################################################################
	# All kernels
	################################################################################################################
	
	Nℓ′ℓs(ℓ′,ℓ,s) = √((2ℓ+1)*(2ℓ′+1)/(4π*(2s+1)))

	function compute_Yℓmatrix_twopoints(ℓ,x1::SphericalPoint,x2::SphericalPoint;
		n_range=-1:1,v_dict=nothing)

		λ,v = read_or_compute_Jy_eigen(v_dict,ℓ)
		# λ,v = Jy_eigen(ℓ)
		dℓ = djmatrix(ℓ,x1,n_range=n_range,λ=λ,v=v)
		Y1 = Ylmatrix(dℓ,ℓ,x1,n_range=n_range,compute_d_matrix=false)
		if x1.θ == x2.θ
			# Reuse d matrix computed at the previous step
			Y2 = Ylmatrix(dℓ,ℓ,x2,n_range=n_range,compute_d_matrix=false)
		else
			# Need to recompute the d matrix, use the eigenvectors computed above
			Y2 = Ylmatrix(dℓ,ℓ,x2,n_range=n_range,λ=λ,v=v,compute_d_matrix=true)
		end
		return Y1,Y2
	end

	function update_Yℓ′_arrays!(Yℓ′_n1_arr,Yℓ′_n2_arr,x1,x2,
		ℓ,ℓ_prev,ℓ_start,ℓ′_range;n_range=-1:1,v_dict=nothing)

		ℓ′_last = last(ℓ′_range)
		ℓ′_first = first(ℓ′_range)

		if ℓ != ℓ_start && ℓ == ℓ_prev
			# re-use previously computed arrays
			# nothing to do here
		elseif ℓ - ℓ_prev == 1
    		# roll the Yℓ′ arrays
    		for ind in ℓ′_first-ℓ:ℓ′_last-ℓ-1
    			@. Yℓ′_n1_arr[ind] = Yℓ′_n1_arr[ind+1]
    		end

    		for ind in ℓ′_first-ℓ:ℓ′_last-ℓ-1
    			@. Yℓ′_n2_arr[ind] = Yℓ′_n2_arr[ind+1]
    		end

    		# compute the last element which is new

    		Y1,Y2 = compute_Yℓmatrix_twopoints(ℓ′_last,x1,x2,n_range=n_range,v_dict=v_dict)
			Yℓ′_n1_arr[ℓ′_last-ℓ][axes(Y1)...] = Y1
    		Yℓ′_n2_arr[ℓ′_last-ℓ][axes(Y2)...] = Y2
    	else
    		# re-initialize the Yℓ′ arrays
    		for ℓ′ in ℓ′_range
    			Y1,Y2 = compute_Yℓmatrix_twopoints(ℓ′,x1,x2,n_range=n_range,v_dict=v_dict)
    			Yℓ′_n1_arr[ℓ′-ℓ][axes(Y1)...] = Y1
    			Yℓ′_n2_arr[ℓ′-ℓ][axes(Y2)...] = Y2
    		end
    	end
	end

	function compute_BiPoSH_without_los(Y12,Y21,x1,x2,ℓ,ℓ′,s_max;
		Yℓ_n1=nothing,Yℓ_n2=nothing,Yℓ′_n1=nothing,Yℓ′_n2=nothing)

		# ignore line-of-sight projection, just the β=γ=0 component
    	# check if (ℓ′,ℓ) has already been computed
    	# We can use the fact that Yℓ′ℓs0(n2,n1) = (-1)^(ℓ+ℓ′+s)Yℓℓ′s0(n1,n2)
    	# If we have computed Yℓℓ′s0(n1,n2) already in the previous step,
		# then we can simply read it in from the dictionary

		if haskey(Y12,(ℓ′,ℓ))
    		Yℓ′ℓ_s0_n1n2 = Y12[(ℓ′,ℓ)]
    	elseif haskey(Y21,(ℓ,ℓ′))
    		Yℓ′ℓ_s0_n1n2 = Y21[(ℓ,ℓ′)] .* ((-1)^(ℓ+ℓ′+s) for s in axes(Y21[(ℓ,ℓ′)],1))
    	else
    		# compute and store it
    		Yℓ′ℓ_s0_n1n2 = BiPoSH_s0(ℓ′,ℓ,1:s_max,0,0,x1,x2,
    						Y_ℓ₁=Yℓ′_n1,Y_ℓ₂=Yℓ_n2)
    		Y12[(ℓ′,ℓ)] = Yℓ′ℓ_s0_n1n2
    	end
    	
    	if haskey(Y21,(ℓ′,ℓ))
    		Yℓ′ℓ_s0_n2n1 = Y21[(ℓ′,ℓ)]
    	elseif haskey(Y12,(ℓ,ℓ′))
    		Yℓ′ℓ_s0_n2n1 = Y12[(ℓ,ℓ′)].* ((-1)^(ℓ+ℓ′+s) for s in axes(Y12[(ℓ,ℓ′)],1))
    	else
    		Yℓ′ℓ_s0_n2n1 = BiPoSH_s0(ℓ′,ℓ,1:s_max,0,0,x2,x1,
    						Y_ℓ₁=Yℓ′_n2,Y_ℓ₂=Yℓ_n1)
    		Y21[(ℓ′,ℓ)] = Yℓ′ℓ_s0_n2n1
    	end

    	return Yℓ′ℓ_s0_n1n2,Yℓ′ℓ_s0_n2n1
	end

	function read_or_compute_Jy_eigen(v_dict,ℓ)
		if !isnothing(v_dict) && haskey(v_dict,ℓ)
   			v = v_dict[ℓ]
   			λ = Float64.(-ℓ:ℓ)
   		else
   			λ,v = Jy_eigen(ℓ)
   			if !isnothing(v_dict) 
   				v_dict[ℓ] = v
   			end
   		end
   		return λ,v
	end

	function assign_Gfn_components!(G_Re_Im::Array{Float64},G::OffsetArray{ComplexF64})
	      ax_offset = CartesianIndices(axes(G)[2:end])
	      ax_flat = CartesianIndices(axes(G_Re_Im)[3:end])
	      assign_Gfn_components!(G_Re_Im,G,ax_flat,ax_offset)
	end

	function assign_Gfn_components!(G_Re_Im::Array{Float64},G::OffsetArray{ComplexF64},
	      ax_flat,ax_offset)

	      for (ind_offset,ind_flat) in zip(ax_offset,ax_flat)
	              @. G[:,ind_offset] = G_Re_Im[:,1,ind_flat] + im*G_Re_Im[:,2,ind_flat]
	      end
	end

	function flow_axisymmetric_without_los(x1::Point3D,x2::Point3D,s_max;
		K_components=-1:1,kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_x1 = Gfn_path_from_source_radius(x1)
		Gfn_path_x2 = Gfn_path_from_source_radius(x2)

		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs dω ν_start_zeros

		num_procs_x1 = load(joinpath(Gfn_path_x1,"parameters.jld2"),"num_procs")
		num_procs_x2 = load(joinpath(Gfn_path_x2,"parameters.jld2"),"num_procs")

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
		modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)
		
		h_ω_arr = hω(x1,x2;r_src=r_src,kwargs...)[ν_start_zeros .+ ν_ind_range] # only in range

		r₁_ind = argmin(abs.(r .- x1.r))
		r₂_ind = argmin(abs.(r .- x2.r))

		obs_at_same_height = r₁_ind == r₂_ind

		function summodes(ℓ_ωind_iter_on_proc)
		
			ℓ_ωind_iter_on_proc = collect(ℓ_ωind_iter_on_proc)
			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			# Get a list of all modes that will be accessed.
			# This can be used to open the fits files before the loops begin.
			# This will cut down on FITS IO costs

			# Gℓ′ω(r,robs) files
			first_mode = first(ℓ_ωind_iter_on_proc)
			last_mode = last(ℓ_ωind_iter_on_proc)
			ℓ′_min_first_mode = max(minimum(ℓ_arr),abs(first_mode[1]-s_max))
			ℓ′_max_last_mode = min(maximum(ℓ_arr),last_mode[1]+s_max)
			modes_minmax = minmax_from_split_array(ℓ_ωind_iter_on_proc)
			ℓ_max_proc = modes_minmax.ℓ_max
			ℓ′_max_proc =  ℓ_max_proc + s_max
			
			proc_id_min_G_x1 = get_processor_id_from_split_array(ℓ_arr,1:Nν_Gfn,
									(ℓ′_min_first_mode,first_mode[2]),num_procs_x1)
			proc_id_max_G_x1 = get_processor_id_from_split_array(ℓ_arr,1:Nν_Gfn,
									(ℓ′_max_last_mode,last_mode[2]),num_procs_x1)

			proc_id_min_G_x2 = get_processor_id_from_split_array(ℓ_arr,1:Nν_Gfn,
                                  	(ℓ′_min_first_mode,first_mode[2]),num_procs_x2)
           	proc_id_max_G_x2 = get_processor_id_from_split_array(ℓ_arr,1:Nν_Gfn,
									(ℓ′_max_last_mode,last_mode[2]),num_procs_x2)

			Gfn_fits_files_obs1 = Gfn_fits_files(Gfn_path_x1,proc_id_min_G_x1:proc_id_max_G_x1)
			Gfn_fits_files_obs2 = Gfn_fits_files(Gfn_path_x2,proc_id_min_G_x2:proc_id_max_G_x2)

			K = zeros(nr,minimum(K_components):maximum(K_components),1:s_max)

			Gsrc = zeros(ComplexF64,nr,0:1)
			drGsrc = zeros(ComplexF64,nr,0:1)
			G_x1 = zeros(ComplexF64,nr,0:1)
			G_x2 = zeros(ComplexF64,nr,0:1)

			# temporary array to precompute the radial part, indices are (r,η)
			f_radial_0_r₁ = zeros(ComplexF64,nr,0:1) # the η=-1 and η=1 terms are identical
			f_radial_1_r₁ = zeros(ComplexF64,nr,-1:1)
			fℓ′ℓsω_r₁ = zeros(ComplexF64,nr,0:1)

			f_radial_0_r₂ = zeros(ComplexF64,nr,0:1) # the η=-1 and η=1 terms are identical
			f_radial_1_r₂ = zeros(ComplexF64,nr,-1:1)
			fℓ′ℓsω_r₂ = zeros(ComplexF64,nr,0:1)

			# Clebsch Gordan coefficients, indices are (s,η,t)
			Cℓ′ℓ = zeros(-1:1,1:s_max,-1:0)

			# cache Ylmn arrays to speed up computation of BiPoSH_s0
			# Create an OffsetArray of the Ylmn OffsetArrays
			T = OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
			Yℓ′_n1_arr = OffsetArray{T}(undef,-s_max:s_max)
    		Yℓ′_n2_arr = OffsetArray{T}(undef,-s_max:s_max)
    		# The array indices will be ℓ′-ℓ values to ensure that we can 
    		# overwrite the preallocated arrays
    		
    		for ind in eachindex(Yℓ′_n1_arr)
    			Yℓ′_n1_arr[ind] = zeros(ComplexF64,-ℓ′_max_proc:ℓ′_max_proc,0:0)
    		end

    		for ind in eachindex(Yℓ′_n2_arr)
    			Yℓ′_n2_arr[ind] = zeros(ComplexF64,-ℓ′_max_proc:ℓ′_max_proc,0:0)
    		end

			# Cache bipolar spherical harmonics in a dict on each worker
			Y12 = Dict{NTuple{2,Int64},OffsetArray{ComplexF64,3,Array{ComplexF64,3}}}()
			Y21 = Dict{NTuple{2,Int64},OffsetArray{ComplexF64,3,Array{ComplexF64,3}}}()

			v_dict = Dict{Int64,OffsetArray{ComplexF64,2,Array{ComplexF64,2}}}()

			# keep track of ℓ to cache Yℓ′ by rolling arrays
			# if ℓ changes by 1 arrays can be rolled
			# if the ℓ wraps back then δℓ will be negative. 
			# In this case we need to recompute the Yℓ′ arrays
			# Start by sorting the modes on the processor so that 
			# the ℓ's are in order

			ℓ_ωind_iter_on_proc = sort(ℓ_ωind_iter_on_proc,by=x->x[1])

			ℓ_prev = first_mode[1]

			# Loop over the Greenfn files
			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_src_Gfn_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_src_Gfn_file)

					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_src_Gfn_file,(ℓ,ω_ind))

					ω = dω*(ν_start_zeros + ω_ind)
		    		
		    		G = read(Gsrc_file[1],:,:,1:2,1,1,ℓω_index_Gsrc_file)
		    		assign_Gfn_components!(G,Gsrc)
		    		G_r₁_rsrc = Gsrc[r₁_ind,0]
		    		G_r₂_rsrc = Gsrc[r₂_ind,0]

		    		if 0 in K_components
		    			G = read(Gsrc_file[1],:,:,1:2,1,2,ℓω_index_Gsrc_file)
		    			assign_Gfn_components!(G,drGsrc)
		    		end

		    		G = nothing

		    		ℓ′_range = intersect(ℓ_arr,abs(ℓ-s_max):ℓ+s_max)

		    		# Precompute Ylmatrix to speed up evaluation of BiPoSH_s0
		    		Yℓ_n1,Yℓ_n2 = compute_Yℓmatrix_twopoints(ℓ,x1,x2,n_range=0:0,v_dict=v_dict)

		    		update_Yℓ′_arrays!(Yℓ′_n1_arr,Yℓ′_n2_arr,x1,x2,
							ℓ,ℓ_prev,first_mode[1],ℓ′_range,n_range=0:0)

			    	ℓ_prev=ℓ

				    for ℓ′ in ℓ′_range

				    	Yℓ′ℓ_s0_n1n2,Yℓ′ℓ_s0_n2n1 = compute_BiPoSH_without_los(
				    								Y12,Y21,x1,x2,ℓ,ℓ′,s_max;
				    								Yℓ_n1=Yℓ_n1,Yℓ_n2=Yℓ_n2,
				    								Yℓ′_n1=Yℓ′_n1_arr[ℓ′-ℓ],
				    								Yℓ′_n2=Yℓ′_n2_arr[ℓ′-ℓ])

			    		# Compute the CG coefficients that appear in fℓ′ℓsω
			    		for t=-1:0,s in 1:s_max,η=-1:1
			    			if (ℓ′<abs(η)) || (ℓ<abs(η+t))
			    				Cℓ′ℓ[η,s,t] = 0
			    			else
			    				Cℓ′ℓ[η,s,t] = clebschgordan(ℓ′,-η,ℓ,η+t,s,t)
			    			end
			    		end

			    		proc_id_mode_G_x1,ℓ′ω_index_G_x1_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ′,ω_ind),num_procs_x1)
			    		proc_id_mode_G_x2,ℓ′ω_index_G_x2_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ′,ω_ind),num_procs_x2)
						
	    				# Green functions based at the observation point for ℓ′
	    				G = read(Gfn_fits_files_obs1[proc_id_mode_G_x1][1],:,:,1:2,1,1,ℓ′ω_index_G_x1_file)
			    		assign_Gfn_components!(G,G_x1)

			    		if !obs_at_same_height
		    				G = read(Gfn_fits_files_obs2[proc_id_mode_G_x2][1],:,:,1:2,1,1,ℓ′ω_index_G_x2_file)
		    				assign_Gfn_components!(G,G_x2)
				    	end

			    		G = nothing

			    		# precompute the radial term in f, α=0
			    		if 0 in K_components
				    		for η=0:1
				    			@. f_radial_0_r₁[:,η] = (-1)^η * G_x1[:,η] * drGsrc[:,η]
				    			if !obs_at_same_height
				    				@. f_radial_0_r₂[:,η] = (-1)^η * G_x2[:,η] * drGsrc[:,η]
				    			end
				    		end
			    		end

			    		# α=1
			    		if (-1 in K_components) || (1 in K_components)
				    		for η=-1:1
				    			@. f_radial_1_r₁[:,η] = (-1)^η * 1/r * G_x1[:,abs(η)] * 
				    				( Ω(ℓ,η)*Gsrc[:,abs(η)] - ((η != -1) ? Gsrc[:,abs(η-1)] : 0) )
				    			if !obs_at_same_height
				    				@. f_radial_1_r₂[:,η] = (-1)^η * 1/r * G_x2[:,abs(η)] * 
				    				( Ω(ℓ,η)*Gsrc[:,abs(η)] - ((η != -1) ? Gsrc[:,abs(η-1)] : 0) )
				    			end
				    		end
				    	end

			    		for s in LinearIndices(Yℓ′ℓ_s0_n2n1)
			    			# radial component (for all s)
			    			
			    			if 0 in K_components
			    				fℓ′ℓsω_r₁[:,0] .= sum(f_radial_0_r₁[:,abs(η)]*Cℓ′ℓ[η,s,0] for η=-1:1)
			    				if !obs_at_same_height
			    					fℓ′ℓsω_r₂[:,0] .= sum(f_radial_0_r₂[:,abs(η)]*Cℓ′ℓ[η,s,0] for η=-1:1)
			    				else
			    					fℓ′ℓsω_r₂[:,0] .= fℓ′ℓsω_r₁[:,0]
			    				end
			    			end

			    			if (-1 in K_components) || (1 in K_components)
			    				fℓ′ℓsω_r₁[:,1] .= sum(f_radial_1_r₁[:,η]*Cℓ′ℓ[η,s,-1] for η=-1:1)
			    				if !obs_at_same_height
			    					fℓ′ℓsω_r₂[:,1] .= sum(f_radial_1_r₂[:,η]*Cℓ′ℓ[η,s,-1] for η=-1:1)
			    				else
			    					fℓ′ℓsω_r₂[:,1] .= fℓ′ℓsω_r₁[:,1]
			    				end
			    			end

			    			if isodd(ℓ+ℓ′+s) && -1 in K_components
			    				# tangential - component i(K⁺ - K⁻), only for odd l+l′+s
								# extra factor of 2 from the (1 - (-1)^(ℓ+ℓ′+s)) term
								@. K[:,-1,s] += -2 * dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
					     					2real(conj(h_ω_arr[ω_ind]) *
					     					(conj(fℓ′ℓsω_r₁[:,1])*G_r₂_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
					     						+ conj(G_r₁_rsrc)*fℓ′ℓsω_r₂[:,1]*Yℓ′ℓ_s0_n2n1[s]) )
			    			end

			    			if 0 in K_components
						     	@. K[:,0,s] +=  dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
						     					2real(conj(h_ω_arr[ω_ind]) *
						     					(conj(fℓ′ℓsω_r₁[:,0])*G_r₂_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
						     						+ conj(G_r₁_rsrc)*fℓ′ℓsω_r₂[:,0]*Yℓ′ℓ_s0_n2n1[s]) )
						    end
							
							if iseven(ℓ+ℓ′+s) && 1 in K_components
								# tangential + component (K⁺ + K⁻), only for even l+l′+s
								# extra factor of 2 from the (1 + (-1)^(ℓ+ℓ′+s)) term
								@. K[:,1,s] +=  2 * dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
					     					2real(conj(h_ω_arr[ω_ind]) *
					     					(conj(fℓ′ℓsω_r₁[:,1])*G_r₂_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
					     						+ conj(G_r₁_rsrc)*fℓ′ℓsω_r₂[:,1]*Yℓ′ℓ_s0_n2n1[s]) )
							end
					    end
					end
				end
			end

			@. K *=  2r^2 * ρ

			close.(values(Gfn_fits_files_src))
			close.(values(Gfn_fits_files_obs1))
			close.(values(Gfn_fits_files_obs2))
			return K
		end

		T = OffsetArray{Float64,3,Array{Float64,3}} # type of arrays to be added to the channels
		return pmapsum(T,summodes,modes_iter)
	end

	function flow_axisymmetric_without_los(n1::Point2D,n2::Point2D,s_max;
		K_components=-1:1,kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		r_obs = get(kwargs,:r_obs,r_obs_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		Gfn_path_obs = Gfn_path_from_source_radius(r_obs)

		@load joinpath(Gfn_path_src,"parameters.jld2") Nν_Gfn ℓ_arr num_procs dω ν_start_zeros

		num_procs_obs = load(joinpath(Gfn_path_obs,"parameters.jld2"),"num_procs")

		ℓ_range = get(kwargs,:ℓ_range,ℓ_arr)
		ν_ind_range = get(kwargs,:ν_ind_range,1:Nν_Gfn)
		modes_iter = Base.Iterators.product(ℓ_range,ν_ind_range)
		
		h_ω_arr = hω(n1,n2;kwargs...)[ν_start_zeros .+ ν_ind_range] # only in range

		r_obs_ind = argmin(abs.(r .- r_obs))

		function summodes(ℓ_ωind_iter_on_proc)

			ℓ_ωind_iter_on_proc = collect(ℓ_ωind_iter_on_proc)
			proc_id_range_Gsrc = get_processor_range_from_split_array(ℓ_arr,1:Nν_Gfn,ℓ_ωind_iter_on_proc,num_procs)
			Gfn_fits_files_src = Gfn_fits_files(Gfn_path_src,proc_id_range_Gsrc)

			# Get a list of all modes that will be accessed.
			# This can be used to open the fits files before the loops begin.
			# This will cut down on FITS IO costs

			# Gℓ′ω(r,robs) files
			first_mode = first(ℓ_ωind_iter_on_proc)
			last_mode = last(ℓ_ωind_iter_on_proc)
			ℓ′_min_first_mode = max(minimum(ℓ_arr),abs(first_mode[1]-s_max))
			ℓ′_max_last_mode = min(maximum(ℓ_arr),last_mode[1]+s_max)
			modes_minmax = minmax_from_split_array(ℓ_ωind_iter_on_proc)
			ℓ_max_proc = modes_minmax.ℓ_max
			ℓ′_max_proc =  ℓ_max_proc + s_max
			
			proc_id_min_Gobs = get_processor_id_from_split_array(ℓ_arr,1:Nν_Gfn,
									(ℓ′_min_first_mode,first_mode[2]),num_procs_obs)
			proc_id_max_Gobs = get_processor_id_from_split_array(ℓ_arr,1:Nν_Gfn,
									(ℓ′_max_last_mode,last_mode[2]),num_procs_obs)

			Gfn_fits_files_obs = Gfn_fits_files(Gfn_path_obs,proc_id_min_Gobs:proc_id_max_Gobs)

			K = zeros(nr,minimum(K_components):maximum(K_components),1:s_max)

			Gsrc = zeros(ComplexF64,nr,0:1)
			drGsrc = zeros(ComplexF64,nr,0:1)
			Gobs = zeros(ComplexF64,nr,0:1)

			# temporary array to precompute the radial part, indices are (r,η)
			f_radial_0_robs = zeros(ComplexF64,nr,0:1) # the η=-1 and η=1 terms are identical
			f_radial_1_robs = zeros(ComplexF64,nr,-1:1)
			fℓ′ℓsω_robs = zeros(ComplexF64,nr,0:1)

			# Clebsch Gordan coefficients, indices are (s,η,t)
			Cℓ′ℓ = zeros(-1:1,1:s_max,-1:0)

			# Cache Ylmn arrays to speed up computation of BiPoSH_s0
			# Create an OffsetArray of the Ylmn OffsetArrays
			T = OffsetArray{ComplexF64,2,Array{ComplexF64,2}}
			Yℓ′_n1_arr = OffsetArray{T}(undef,-s_max:s_max)
   			Yℓ′_n2_arr = OffsetArray{T}(undef,-s_max:s_max)
  			# The array indices will be ℓ′-ℓ values to ensure that we can 
  			# overwrite the preallocated arrays
  			  
   			for ind in eachindex(Yℓ′_n1_arr)
   				Yℓ′_n1_arr[ind] = zeros(ComplexF64,-ℓ′_max_proc:ℓ′_max_proc,0:0)
   			end

   			for ind in eachindex(Yℓ′_n2_arr)
   				Yℓ′_n2_arr[ind] = zeros(ComplexF64,-ℓ′_max_proc:ℓ′_max_proc,0:0)
   			end

			# Cache bipolar spherical harmonics in a dict on each worker
			Y12 = Dict{NTuple{2,Int64},OffsetArray{ComplexF64,3,Array{ComplexF64,3}}}()
			Y21 = Dict{NTuple{2,Int64},OffsetArray{ComplexF64,3,Array{ComplexF64,3}}}()

			v_dict = Dict{Int64,OffsetArray{ComplexF64,2,Array{ComplexF64,2}}}()

			# keep track of ℓ to cache Yℓ′ by rolling arrays
			# if ℓ changes by 1 arrays can be rolled
			# if the ℓ wraps back then δℓ will be negative. 
			# In this case we need to recompute the Yℓ′ arrays
			# Start by sorting the modes on the processor so that 
			# the ℓ's are in order

			ℓ_ωind_iter_on_proc = sort(ℓ_ωind_iter_on_proc,by=x->x[1])
			ℓ_prev = first_mode[1]

			# Loop over the Greenfn files
			for (proc_id,Gsrc_file) in Gfn_fits_files_src

				# Get a list of (ℓ,ω) in this file
				modes_in_src_Gfn_file = split_product_across_processors(ℓ_arr,1:Nν_Gfn,num_procs,proc_id)

				for (ℓ,ω_ind) in intersect(ℓ_ωind_iter_on_proc,modes_in_src_Gfn_file)

					ℓω_index_Gsrc_file = get_index_in_split_array(modes_in_src_Gfn_file,(ℓ,ω_ind))

					ω = dω*(ν_start_zeros + ω_ind)

		 	 		G = read(Gsrc_file[1],:,:,1:2,1,1,ℓω_index_Gsrc_file)
					assign_Gfn_components!(G,Gsrc)
		 	 		G_robs_rsrc = Gsrc[r_obs_ind,0]
	
		 	 		if 0 in K_components
		 	 			G = read(Gsrc_file[1],:,:,1:2,1,2,ℓω_index_Gsrc_file)
		 	 			assign_Gfn_components!(G,drGsrc)
		 	   		end
	
		 	   		G = nothing

		 	   		ℓ′_range = intersect(ℓ_arr,abs(ℓ-s_max):ℓ+s_max)
	
		 	   		# Precompute Ylmatrix to speed up evaluation of BiPoSH_s0

		 	   		Yℓ_n1,Yℓ_n2 = compute_Yℓmatrix_twopoints(ℓ,n1,n2,n_range=0:0,v_dict=v_dict)

		 	   		update_Yℓ′_arrays!(Yℓ′_n1_arr,Yℓ′_n2_arr,n1,n2,
							ℓ,ℓ_prev,first_mode[1],ℓ′_range,n_range=0:0)

			    	ℓ_prev=ℓ

				    for ℓ′ in ℓ′_range

				    	Yℓ′ℓ_s0_n1n2,Yℓ′ℓ_s0_n2n1 = compute_BiPoSH_without_los(
				    								Y12,Y21,n1,n2,ℓ,ℓ′,s_max;
				    								Yℓ_n1=Yℓ_n1,Yℓ_n2=Yℓ_n2,
				    								Yℓ′_n1=Yℓ′_n1_arr[ℓ′-ℓ],
				    								Yℓ′_n2=Yℓ′_n2_arr[ℓ′-ℓ])

			    		# Compute the CG coefficients that appear in fℓ′ℓsω
			    		for t=-1:0,s in 1:s_max,η=-1:1
			    			if (ℓ′<abs(η)) || (ℓ<abs(η+t))
			    				Cℓ′ℓ[η,s,t] = 0
			    			else
			    				Cℓ′ℓ[η,s,t] = clebschgordan(ℓ′,-η,ℓ,η+t,s,t)
			    			end
			    		end

			    		proc_id_mode_Gobs,ℓ′ω_index_Gobs_file = procid_and_mode_index(ℓ_arr,1:Nν_Gfn,(ℓ′,ω_ind),num_procs_obs)
			    								
		  		  		# Green functions based at the observation point for ℓ′
		  		  		G = read(Gfn_fits_files_obs[proc_id_mode_Gobs][1],:,:,1:2,1,1,ℓ′ω_index_Gobs_file)
			    		assign_Gfn_components!(G,Gobs)

			    		G = nothing

			    		# precompute the radial term in f, α=0
			    		if 0 in K_components
				    		for η=0:1
				    			@. f_radial_0_robs[:,η] = (-1)^η * Gobs[:,η] * drGsrc[:,η]
				    		end
			    		end

			    		# α=1
			    		if (-1 in K_components) || (1 in K_components)
				    		for η=-1:1
				    			@. f_radial_1_robs[:,η] = (-1)^η * 1/r * Gobs[:,abs(η)] * 
				    				( Ω(ℓ,η)*Gsrc[:,abs(η)] - ((η != -1) ? Gsrc[:,abs(η-1)] : 0) )
				    		end
				    	end

			    		for s in LinearIndices(Yℓ′ℓ_s0_n2n1)
			    			# radial component (for all s)
			    			
			    			if 0 in K_components
			    				fℓ′ℓsω_robs[:,0] .= sum(f_radial_0_robs[:,abs(η)]*Cℓ′ℓ[η,s,0] for η=-1:1) 
			    			end

			    			if (-1 in K_components) || (1 in K_components)
			    				fℓ′ℓsω_robs[:,1] .= sum(f_radial_1_robs[:,η]*Cℓ′ℓ[η,s,-1] for η=-1:1)
			    			end

			    			if isodd(ℓ+ℓ′+s) && -1 in K_components
			    				# tangential - component i(K⁺ - K⁻), only for odd l+l′+s
								# extra factor of 2 from the (1 - (-1)^(ℓ+ℓ′+s)) term
								@. K[:,-1,s] += -2 * dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
					     					2real(conj(h_ω_arr[ω_ind]) *
					     					(conj(fℓ′ℓsω_robs[:,1])*G_robs_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
					     						+ conj(G_robs_rsrc)*fℓ′ℓsω_robs[:,1]*Yℓ′ℓ_s0_n2n1[s]) )
			    			end

			    			if 0 in K_components
						     	@. K[:,0,s] +=  dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
						     					2real(conj(h_ω_arr[ω_ind]) *
						     					(conj(fℓ′ℓsω_robs[:,0])*G_robs_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
						     						+ conj(G_robs_rsrc)*fℓ′ℓsω_robs[:,0]*Yℓ′ℓ_s0_n2n1[s]) )
						    end
							
							if iseven(ℓ+ℓ′+s) && 1 in K_components
								# tangential + component (K⁺ + K⁻), only for even l+l′+s
								# extra factor of 2 from the (1 + (-1)^(ℓ+ℓ′+s)) term
								@. K[:,1,s] +=  2 * dω/2π * ω^3 * Powspec(ω) * Nℓ′ℓs(ℓ′,ℓ,s) * 
					     					2real(conj(h_ω_arr[ω_ind]) *
					     					(conj(fℓ′ℓsω_robs[:,1])*G_robs_rsrc*conj(Yℓ′ℓ_s0_n1n2[s]) 
					     						+ conj(G_robs_rsrc)*fℓ′ℓsω_robs[:,1]*Yℓ′ℓ_s0_n2n1[s]) )
							end
					    end
					end
				end
			end

			@. K *= 2r^2 * ρ

			close.(values(Gfn_fits_files_obs))
			close.(values(Gfn_fits_files_src))

			return K
		end

		T = OffsetArray{Float64,3,Array{Float64,3}} # type of arrays to be added to the channels
		return pmapsum(T,summodes,modes_iter)
	end

	function meridional_flow_ψ_srange(x1,x2,s_max;kwargs...)
		Kv = flow_axisymmetric_without_los(x1,x2,s_max;K_components=0:1,kwargs...)

		Kψ_imag = zeros(nr,1:s_max)

		@inbounds for s in axes(Kψ_imag,2)
			Kψ_imag[:,s] .= ddr*(Kv[:,1,s]./ρ) .+ @. Kv[:,1,s]/(ρ*r) - 2*Ω(s,0)*Kv[:,0,s]/(ρ*r)
		end

		return Kψ_imag
	end
end

module traveltimes
	using Main.kernel

	uniform_rotation_uplus(Ω_rot=20e2/Rsun) = @. √(4π/3)*im*Ω_rot*r


	function δτ_uniform_rotation_firstborn_int_K_u(x1,x2;Ω_rot=20e2/Rsun,kwargs...)
		K₊ = kernel_uniform_rotation_uplus(x1,x2;kwargs...)
		u⁺ = uniform_rotation_uplus(Ω_rot)

		δτ = real.(simps((@. K₊*imag(u⁺)),r))
	end

	function δτ_uniform_rotation_firstborn_int_hω_δCω(n1,n2;Ω_rot=20e2/Rsun,bounce_no=1,kwargs...)
		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dω ν_start_zeros Nν_Gfn

		h_ω = get(kwargs,:hω,nothing)
		if isnothing(h_ω)
			h_ω = hω(n1,n2;bounce_no=bounce_no,kwargs...)
		end
		h_ω = h_ω[ν_start_zeros .+ (1:Nν_Gfn)] # only in range

		δCω = δCω_uniform_rotation_firstborn_integrated_over_angle(n1,n2;
			Ω_rot=Ω_rot,kwargs...)[ν_start_zeros .+ (1:Nν_Gfn)]

		δτ = simps((@. 2real(conj(h_ω)*δCω)),dω/2π)
	end

	function δτ_uniform_rotation_rotatedframe_int_hω_δCω_linearapprox(n1::Point2D,n2::Point2D;
		Ω_rot=20e2/Rsun,bounce_no=1,kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dω ν_start_zeros Nν_Gfn

		h_ω = get(kwargs,:hω,nothing)
		if isnothing(h_ω)
			h_ω = hω(n1,n2;bounce_no=bounce_no,kwargs...)
		end
		h_ω = h_ω[ν_start_zeros .+ (1:Nν_Gfn)] # only in range

		δCω = δCω_uniform_rotation_rotatedwaves_linearapprox(n1,n2;
			Ω_rot=Ω_rot,kwargs...)[ν_start_zeros .+ (1:Nν_Gfn)]

		δτ = simps((@. 2real(conj(h_ω)*δCω)),dω/2π)
	end

	function δτ_uniform_rotation_rotatedframe_int_ht_δCt(n1::Point2D,n2::Point2D;
		Ω_rot=20e2/Rsun,bounce_no=1,kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dt Nt dν

		Cω_n1n2 = Cω(n1,n2;kwargs...)
		Ct_n1n2 = Ct(Cω_n1n2,dν)
		τ_ind_arr = time_window_indices_by_fitting_bounce_peak(Ct_n1n2,n1,n2,dt=dt,Nt=Nt,bounce_no=bounce_no)

		h_ω = get(kwargs,:hω,nothing)
		h_t = get(kwargs,:ht, !isnothing(h_ω) ? @fft_ω_to_t(h_ω) :
			ht(Cω_n1n2,n1,n2;τ_ind_arr=τ_ind_arr,kwargs...))
		h_t = h_t[τ_ind_arr]

		δC_t = δCt_uniform_rotation_rotatedwaves(n1,n2;Ω_rot=Ω_rot,τ_ind_arr=τ_ind_arr,kwargs...)

		δτ = simps((h_t.*δC_t.parent),dt)
	end

	function δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1::Point2D,n2::Point2D;
		Ω_rot=20e2/Rsun,bounce_no=1,τ_ind_arr=nothing,kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dt dν Nt

		if isnothing(get(kwargs,:Cω,nothing)) && isnothing(get(kwargs,:∂ϕ₂Cω,nothing))
			Cω_n1n2,∂ϕ₂Cω_n1n2 = Cω_∂ϕ₂Cω(n1,n2;kwargs...)
		elseif isnothing(get(kwargs,:Cω,nothing))
			Cω_n1n2 = Cω(n1,n2;kwargs...)
		elseif isnothing(get(kwargs,:∂ϕ₂Cω,nothing))
			∂ϕ₂Cω_n1n2 = ∂ϕ₂Cω(n1,n2;kwargs...)
		end

		if isnothing(τ_ind_arr)
			Ct_n1n2 = Ct(Cω_n1n2,dν)
			τ_ind_arr = time_window_indices_by_fitting_bounce_peak(Ct_n1n2,
								n1,n2,dt=dt,Nt=Nt,bounce_no=bounce_no)
		end

		h_ω = get(kwargs,:hω,nothing)
		h_t = get(kwargs,:ht, !isnothing(h_ω) ? @fft_ω_to_t(h_ω) :
			ht(Cω_n1n2,n1,n2;τ_ind_arr=τ_ind_arr,kwargs...))

		∂ϕ₂Ct_n1n2 = ∂ϕ₂Ct(∂ϕ₂Cω_n1n2,dν)
		δC_t = δCt_uniform_rotation_rotatedwaves_linearapprox(∂ϕ₂Ct_n1n2;
											Ω_rot=Ω_rot,kwargs...)
		
		δτ = simps((h_t[τ_ind_arr].*δC_t[τ_ind_arr]),dt)
	end

	function δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1::Point2D,n2_arr::Vector{<:Point2D};
		Ω_rot=20e2/Rsun,bounce_no=1,τ_ind_arr=nothing,kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dt dν Nt

		Cω_n1n2 = get(kwargs,:Cω,Cω(n1,n2_arr;kwargs...))
		∂ϕ₂Cω_n1n2 = get(kwargs,:∂ϕ₂Cω,∂ϕ₂Cω(n1,n2_arr;kwargs...))
		
		if isnothing(τ_ind_arr)
			Ct_n1n2 = Ct(Cω_n1n2,dν)
			τ_ind_arr = time_window_indices_by_fitting_bounce_peak(Ct_n1n2,
								n1,n2_arr,dt=dt,Nt=Nt,bounce_no=bounce_no)
		end

		h_ω = get(kwargs,:hω,nothing)
		h_t = get(kwargs,:ht, !isnothing(h_ω) ? @fft_ω_to_t(h_ω) :
			ht(Cω_n1n2,n1,n2;τ_ind_arr=τ_ind_arr,kwargs...))

		∂ϕ₂Ct_n1n2 = ∂ϕ₂Ct(∂ϕ₂Cω_n1n2,dν)
		δC_t = δCt_uniform_rotation_rotatedwaves_linearapprox(∂ϕ₂Ct_n1n2;
											Ω_rot=Ω_rot,kwargs...)

		δτ = zeros(length(n2_arr))
		for (n2ind,τ_inds) in enumerate(τ_ind_arr)
			δτ[n2ind] = simps(h_t[τ_inds,n2ind].*δC_t[τ_inds,n2ind],dt)
		end
		return δτ
	end

	function validate(n1,n2;kwargs...)

		r_src = get(kwargs,:r_src,r_src_default)
		Gfn_path_src = Gfn_path_from_source_radius(r_src)
		@load joinpath(Gfn_path_src,"parameters.jld2") dt dν

		h_t = ht(n1,n2;kwargs...)
		h_ω = @fft_t_to_ω(h_t)

		δτ1 = δτ_uniform_rotation_firstborn_int_K_u(n1,n2;hω=h_ω,kwargs...)
		@printf "%-50s %g\n" "First Born, ∫dr u(r) K(r)" round(δτ1,sigdigits=3)

		δτ2 = δτ_uniform_rotation_firstborn_int_hω_δCω(n1,n2;hω=h_ω,kwargs...)
		@printf "%-50s %g\n" "First Born, ∫dω/2π h(ω) δC_FB(ω)" round(δτ2,sigdigits=3)

		δτ3 = δτ_uniform_rotation_rotatedframe_int_hω_δCω_linearapprox(n1,n2;hω=h_ω,kwargs...)
		@printf "%-50s %g\n" "Rotated frame, ∫dω/2π h(ω) δC_R_lin(ω)" round(δτ3,sigdigits=3)

		δτ4 = δτ_uniform_rotation_rotatedframe_int_ht_δCt(n1,n2;ht=h_t,kwargs...)
		@printf "%-50s %g\n" "Rotated frame, ∫dt h(t) δC_R(t)" round(δτ4,sigdigits=3)

		δτ5 = δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1,n2;ht=h_t,kwargs...)
		@printf "%-50s %g\n" "Rotated frame, ∫dt h(t) δC_R_lin(t)" round(δτ5,sigdigits=3)
	end
end

module kernel3D
	using Main.kernel
	import WignerD: Ylmatrix

	function flow_longitudinal_slice(x1,x2,s_max;kwargs...)
		
		K′jl0_r = flow_axisymmetric_without_los(x1,x2,s_max;kwargs...)
		Kjl0_r = similar(K′jl0_r)

		@. Kjl0_r[:,0,:] = K′jl0_r[:,0,:]
		# verify these relations
		@. Kjl0_r[:,1,:] = (K′jl0_r[:,1,:] + K′jl0_r[:,-1,:])
		@. Kjl0_r[:,-1,:] = (K′jl0_r[:,1,:] - K′jl0_r[:,-1,:])

		nθ = get(kwargs,:nθ,20) :: Integer
		K_components = get(kwargs,:K_components,-1:1)
		θ_arr = LinRange(0,π,nθ)
		# Slice at ϕ=0

		K2D_r_θ = zeros(ComplexF64,nr,K_components,nθ)

		# The 3D kernel is given by ∑ₛₙ Ks0n(r)* Ys0n(θ,ϕ)
		# For axisymmetric flows the r and θ components of the kernel are real

		for s in 1:s_max
			λ,v = Jy_eigen(s)
			ds_θ = zeros(0:0,-1:1)
			for (θind,θ) in enumerate(θ_arr)
				Ys0n = Ylmatrix(ds_θ,s,(θ,0),
					compute_d_matrix=true,λ=λ,v=v,
					m_range=0:0,n_range=K_components)
				
				@. K2D_r_θ[:,:,θind] += Kjl0_r[:,:,s].*Ys0n[0,:]
			end
		end

		return real.(K2D_r_θ)
	end

	function flow_latitudinal_slice(x1,x2,s_max;kwargs...)
		
		K′jl0_r = flow_axisymmetric_without_los(x1,x2,s_max;kwargs...)
		Kjl0_r = similar(K′jl0_r)

		@. Kjl0_r[:,0,:] = K′jl0_r[:,0,:]
		# verify these relations
		@. Kjl0_r[:,1,:] = (K′jl0_r[:,1,:] + K′jl0_r[:,-1,:])
		@. Kjl0_r[:,-1,:] = (K′jl0_r[:,1,:] - K′jl0_r[:,-1,:])

		nϕ = get(kwargs,:nϕ,20) :: Integer
		K_components = get(kwargs,:K_components,-1:1)
		ϕ_arr = LinRange(0,2π,nϕ)
		# Slice at ϕ=0

		K2D_r_ϕ = zeros(ComplexF64,nr,K_components,nϕ)

		# The 3D kernel is given by ∑ₛₙ Ks0n(r)* Ys0n(θ,ϕ)
		# For axisymmetric flows the r and θ components of the kernel are real

		for s in 1:s_max
			λ,v = Jy_eigen(s)
			ds_θ = djmatrix(s,π/2,n_range=-1:1)
			for (ϕind,ϕ) in enumerate(ϕ_arr)
				Ys0n = Ylmatrix(ds_θ,s,(π/2,ϕ),
					compute_d_matrix=false,m_range=0:0,n_range=K_components)
				
				@. K2D_r_θ[:,:,ϕind] += Kjl0_r[:,:,s].*Ys0n[0,:]
			end
		end

		return real.(K2D_r_θ)
	end

	Kr_longitudinal_slice(x1,x2,s_max;kwargs...) = 
		flow_longitudinal_slice(x1,x2,s_max;kwargs...,K_components=0:0)
	
	Kθ_longitudinal_slice(x1,x2,s_max;kwargs...) = 
		flow_longitudinal_slice(x1,x2,s_max;kwargs...,K_components=1:1)

	Kϕ_longitudinal_slice(x1,x2,s_max;kwargs...) = 
		flow_longitudinal_slice(x1,x2,s_max;kwargs...,K_components=-1:-1)

	Kr_longitudinal_slice(x1,x2,s_max;kwargs...) = 
		flow_latitudinal_slice(x1,x2,s_max;kwargs...,K_components=0:0)
	
	Kθ_longitudinal_slice(x1,x2,s_max;kwargs...) = 
		flow_latitudinal_slice(x1,x2,s_max;kwargs...,K_components=1:1)

	Kϕ_longitudinal_slice(x1,x2,s_max;kwargs...) = 
		flow_latitudinal_slice(x1,x2,s_max;kwargs...,K_components=-1:-1)
end