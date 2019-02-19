@static if VERSION >= v"0.7.0"
	using Distributed
end

@static if VERSION >= v"0.7.0"
	@everywhere begin
			using LinearAlgebra,DelimitedFiles
	end
end

@everywhere using FileIO,JLD2,Printf
import PyPlot
plt = PyPlot


function get_processor_id_from_split_array(arr1,arr2,(arr1_value,arr2_value),num_procs)
	# Find the closest match in arrays
	
	num_tasks = length(arr1)*length(arr2);

	a1_match_index = argmin(abs.(arr1 .- arr1_value))
	a2_match_index = argmin(abs.(arr2 .- arr2_value))

	num_tasks_per_process,num_tasks_leftover = div(num_tasks,num_procs),mod(num_tasks,num_procs)

	proc_id = 1
	num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
	total_tasks_till_proc_id = num_tasks_on_proc

	task_no = 0

	for (ind2,a2) in enumerate(arr2), (ind1,a1) in enumerate(arr1)
		
		task_no +=1
		if task_no > total_tasks_till_proc_id
			proc_id += 1
			num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
			total_tasks_till_proc_id += num_tasks_on_proc
		end

		if ind2< a2_match_index
			continue
		end

		if (ind2 == a2_match_index) && (ind1 == a1_match_index)
			break
		end
	end

	return proc_id
end

struct Gfn
	mode :: NamedTuple{(:ω, :ω_ind, :ℓ),Tuple{Float64,Int64,Int64}}
	α :: Vector{ComplexF64}
	β :: Vector{ComplexF64}
end

function load_green_fn_onefreq(G_path,ν_test)
   	
	@load joinpath(G_path,"parameters.jld2") ν_arr ℓ_arr num_procs
	ω_arr = 2π .* ν_arr
	ν_arr_indices = 1:length(ν_arr)

	ν_test_index = argmin(abs.(ν_arr .- ν_test));

	Gfn_arr_onefreq = Vector{Gfn}(undef,length(ℓ_arr))

	proc_id_start = get_processor_id_from_split_array(ℓ_arr,ν_arr_indices,(ℓ_arr[1],ν_test_index),num_procs);
    proc_id_end = get_processor_id_from_split_array(ℓ_arr,ν_arr_indices,(ℓ_arr[end],ν_test_index),num_procs);

   	Gfn_ℓ_index = 1
    for proc_id in proc_id_start:proc_id_end
    	G_proc_file = joinpath(G_path,@sprintf "Gfn_proc_%03d.jld2" proc_id)
    	@load G_proc_file Gfn_arr
    	for Gfn in Gfn_arr
    		
    		if Gfn.mode[:ω_ind] == ν_test_index
	    		Gfn_arr_onefreq[Gfn_ℓ_index] = Gfn
	    		Gfn_ℓ_index += 1
    		end

    		if Gfn_ℓ_index > length(ℓ_arr)
    			println("Finished loading Green functions from $G_path")
    			# finished loading all Green functions
    			break
    		end
    	end
    end

    @assert(all([isassigned(Gfn_arr_onefreq,ind) for ind in eachindex(Gfn_arr_onefreq)]) ,"Array assignment failed for $G_path")

    return Gfn_arr_onefreq
end

function main()

    ν_test = 3e-3; #mHz

    modelS_meta = readdlm("ModelS.meta",comments=true, comment_char='#')
	Msun,Rsun = modelS_meta[1:2]

	modelS_detailed = readdlm("ModelS.detailed",comments=true, comment_char='#')
	r = modelS_detailed[:,1];
	flip = (length(r)-1):-1:1
	nr = length(flip)
	r = r[flip]
	r_frac = r./Rsun;

	r_src_1 = Rsun*0.8; src_1_index = argmin(abs.(r.-r_src_1));
	r_src_2 = Rsun*0.6; src_2_index = argmin(abs.(r.-r_src_2));

	Gfn_arr_onefreq_src1 = load_green_fn_onefreq("/scratch/jishnu/Greenfn_src0.80Rsun",ν_test);
    Gfn_arr_onefreq_src2 = load_green_fn_onefreq("/scratch/jishnu/Greenfn_src0.60Rsun",ν_test);

	sum1 = 0 #zeros(ComplexF64,nr)
	sum2 = 0 #zeros(ComplexF64,nr)

	for (Gfn1,Gfn2) in zip(Gfn_arr_onefreq_src1,Gfn_arr_onefreq_src2)
		sum1 += Gfn1.α[src_2_index] *(2*Gfn1.mode.ℓ+1)
		sum2 += Gfn2.α[src_1_index] *(2*Gfn2.mode.ℓ+1)
		if Gfn1.mode.ℓ == Gfn2.mode.ℓ
			println("Grr(r2,r1) $(Gfn1.α[src_2_index]) Grr(r1,r2) $(Gfn2.α[src_1_index])")
		end
	end

	println("Σ_ℓ (2ℓ+1)*αℓ(r2,r1) $sum1")
	println("Σ_ℓ (2ℓ+1)*αℓ(r1,r2) $sum2")

end