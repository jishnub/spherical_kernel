using PyPlot,PointsOnASphere,LaTeXStrings,LinearAlgebra,FITSIO,JLD2
using PyCall
@pyimport matplotlib.ticker as ticker

SCRATCH = ENV["SCRATCH"]

function plot_traveltimes_validation(;nϕ=10,ℓ_range=20:100,bounce_no=1)
	ϕ2_arr = collect(LinRange(0.7,1.2,nϕ))
	n1 = Point2D(π/2,0)
	n2_arr = [Point2D(π/2,ϕ2) for ϕ2 in ϕ2_arr]

	# δτ_rot2 = zeros(nϕ)

	δτ_FB = Main.kernel.δτ_uniform_rotation_firstborn_int_K_u(n1,n2_arr,bounce_no=bounce_no,ℓ_range=ℓ_range)
	δτ_rot2 = Main.kernel.δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1,n2_arr,bounce_no=bounce_no,ℓ_range=ℓ_range)
	# for (ind,n2) in enumerate(n2_arr)
	# 	δτ_rot2[ind] = Main.kernel.δτ_uniform_rotation_rotatedframe_int_ht_δCt_linearapprox(n1,n2,bounce_no=bounce_no,ℓ_range=ℓ_range)
	# end

	ϕ2_arr .*= 180/π # convert to degrees
	plot(ϕ2_arr,δτ_rot2,"p-",label="Rotated frame, linearized",lw=0.6)
	plot(ϕ2_arr,δτ_FB,"o--",label="First Born approx")
	xlabel("Distance (degrees)",fontsize=13)
	title("Travel time shifts",fontsize=13)
	legend(loc="best")
	savefig("traveltimes_validation.png")
end

function compute_K10_different_arrival_bounces()
	n1 = Point2D(π/2,0); n2 = Point2D(π/2,π/3)
	kernels = zeros(Main.kernel.nr,3)

	for (ind,bounce_no) in enumerate((1,2,4))
		kernels[:,ind] .= imag.(Main.kernel.kernel_uniform_rotation_uplus(n1,n2,ℓ_range=20:100,bounce_no=bounce_no))
	end

	f = FITS("$SCRATCH/kernels/K10_different_bounces.fits")
	write(f,kernels)
	close(f)

	return kernels
end

function plot_K10_different_arrival_bounces()

	kernels = compute_K10_different_arrival_bounces()

	r = Main.kernel.r
	Rsun = Main.kernel.Rsun
	r ./= Rsun
	plot(r,normalize(kernels[:,1],Inf),label="first bounce")
	plot(r,normalize(kernels[:,2],Inf),label="second bounce")
	plot(r,normalize(kernels[:,3],Inf),label="fourth bounce")

	
	xlim(0.3,r[end])
	xlabel(L"$r/R_\odot$",fontsize=13)
	ylabel(L"Normalized $K_{10}(r)$",fontsize=13)
	legend(loc="best")
end

function plot_time_distance(;t_max=6,t_min=0.3,ℓ_range = 20:100,nϕ=maximum(ℓ_range)) # time in hours
	r_src=Main.crosscov.Rsun-75e5
	Gfn_path_src = Main.Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt Nt

	t_max_ind = Int(floor(t_max*60^2/dt))+1
	t_min_ind = Int(floor(t_min*60^2/dt))+1
	t_inds = t_min_ind:t_max_ind

	t = t_inds.*dt./60^2

	ϕ = LinRange(5,90,nϕ).*π/180

	C = Main.crosscov.CΔϕt(;τ_ind_arr=t_inds,ℓ_range=ℓ_range,Δϕ_arr=ϕ)
	C = copy(transpose(C))
	C60 = Main.crosscov.Ct(π/3,ℓ_range=ℓ_range)[t_inds]

	ax1 = subplot2grid((1,3),(0,0),colspan=2)
	ax2 = subplot2grid((1,3),(0,2),colspan=1)
	ax1[:pcolormesh](ϕ.*180/π,t,C./maximum(abs,C),
		cmap="Greys",vmax=1,vmin=-1,rasterized=true)
	ax1[:set_xlabel]("Angular separation [degrees]",fontsize=12);
	ax1[:set_ylabel]("Time [hours]",fontsize=12);
	ax1[:set_title]("Time-distance diagram",fontsize=12);
	ax1[:axvline](60,color="black",ls="dashed",lw=0.6)

	ax2[:plot](C60,t,lw=0.7)
	ax2[:yaxis][:set_major_formatter](ticker.NullFormatter())
	ax2[:set_ylim](ax1[:get_ylim]())

	tight_layout()
end

function plot_C_spectrum(;ℓ_range=nothing,ν_ind_range=nothing)

	r_src=Main.crosscov.Rsun-75e5
	Gfn_path_src = Main.Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") ν_arr

	Nν_Gfn = length(ν_arr)

	Cωℓ = Main.crosscov.Cωℓ_spectrum(ℓ_range=ℓ_range,ν_ind_range=ν_ind_range)
	ν_ind_range,ℓ_range = axes(Cωℓ)
	ℓ_edges = collect(first(ℓ_range):last(ℓ_range)+1)
	ν_edges = collect(ν_arr[first(ν_ind_range):last(ν_ind_range)]).*1e3
	spec = Cωℓ.parent./maximum(abs,Cωℓ)

	pcolormesh(ℓ_edges,ν_edges,spec,
		cmap="Oranges",vmax=0.5,vmin=0,rasterized=true);
	xlabel(L"$\ell$",fontsize=12);
	ylabel("Frequency [mHz]",fontsize=12);
	title("Spectrum of cross covariance",fontsize=12);
	tight_layout()
end

function plot_Ct_groups_of_modes(;t_max=6,t_min=0.3)
	r_src=Main.crosscov.Rsun-75e5
	Gfn_path_src = Main.Greenfn_radial.Gfn_path_from_source_radius(r_src)
	@load joinpath(Gfn_path_src,"parameters.jld2") dt Nt

	t_max_ind = Int(floor(t_max*60^2/dt))+1
	t_min_ind = Int(floor(t_min*60^2/dt))+1

	t_inds = t_min_ind:t_max_ind
	t = t_inds.*dt/60^2
	nt = length(t)

	n1 = Point2D(π/2,0)
	n2 = Point2D(π/2,π/3)

	ℓ_ranges = Base.Iterators.partition(20:99,20)

	C = zeros(nt,length(ℓ_ranges)+1)

	for (ind,ℓ_range) in enumerate(ℓ_ranges)
		C[:,ind] = Main.crosscov.Ct(n1,n2,ℓ_range=ℓ_range)[t_inds]
	end

	C[:,end] = Main.crosscov.Ct(n1,n2,ℓ_range=20:99)[t_inds]

	fig,ax = subplots(nrows=size(C,2),ncols=1)
	for axis in ax
		axis[:set_xlim](first(t),last(t))
	end

	ax[1][:set_title]("Cross covariance (Δϕ=60 degrees)")

	for (ind,ℓ_range) in enumerate(ℓ_ranges)
		ax[ind][:plot](t,C[:,ind],label="ℓ=$(ℓ_range[1]:ℓ_range[end])")
		ax[ind][:xaxis][:set_major_formatter](ticker.NullFormatter())
		ax[ind][:yaxis][:set_major_locator](ticker.MaxNLocator(3))
		ax[ind][:legend](loc="upper right",bbox_to_anchor=(1.01,1.5))
	end

	ax[end][:plot](t,C[:,end])
	ax[end][:set_title]("Sum over ℓ")
	ax[end][:yaxis][:set_major_locator](ticker.MaxNLocator(3))
	
	xlabel("Time [hours]",fontsize=12)
	tight_layout()
end