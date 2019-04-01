using OffsetArrays

function wigner3j!(w3j::OffsetVector{Float64,Vector{Float64}},
    j2, j3, m1, m2, m3)

    jmin = max(abs(j2-j3), abs(m1)) ::Int
    jmax = (j2 + j3) ::Int

    flag1 = 0
    flag2 = 0

    scalef = 1.0e3

    fill!(w3j,0.0)

    if (abs(m2) > j2 || abs(m3) > j3) || !iszero(m1 + m2 + m3) || (jmax < jmin)
        return
    end

    @assert(first(axes(w3j,1))<=jmin && last(axes(w3j,1))>=jmax,
        "Array not large enough to store all wigner 3j components")

    a(j) = √( (j^2 - (j2 - j3)^2 ) * ( (j2 + j3 + 1)^2 - j^2 ) * ( j^2 - m1^2 ) )::Real
    y(j) = -(2j+1)*(m1*(j2*(j2 + 1) - j3*(j3 + 1))-(m3-m2)*j*(j + 1))

    x(j) = j*a(j+1)
    z(j) = (j+1)*a(j)

    function normw3j!(w3j)
        norm = sum((2j+1)*w3j[j]^2 for j in jmin:jmax)
        @. w3j /= √(norm)::Real
    end

    function fixsign!(w3j)
        begin
        if ( (w3j[jmax] < 0 && iseven(j2-j3+m2+m3)) || 
            (w3j[jmax] > 0 && isodd(j2-j3+m2+m3)) )
            @. w3j *= -1.0
        end
        end
    end

    rs = zeros(jmin:jmax); wl = zeros(jmin:jmax); wu = zeros(jmin:jmax);

    #--------------------------------------------------------------------------
    #
    #   Only one term is present
    #
    #--------------------------------------------------------------------------
        
    
    if jmin == jmax
        w3j[jmin] = 1/√(2jmin + 1)::Real
        
        if ( (w3j[jmin] < 0 && iseven(j2-j3+m2+m3)) || 
            (w3j[jmin] > 0 && isodd(j2-j3+m2+m3)) )

            w3j[jmin] *= -1.0
        end

        return
    end

    #--------------------------------------------------------------------------
    #
    #   Calculate lower non-classical values for [jmin, jn]. If the second term
    #   can not be calculated because the recursion relationsips give rise to a
    #   1/0,  set flag1 to 1.  If all m's are zero,  this is not a
    #   problem as all odd terms must be zero.
    #
    #--------------------------------------------------------------------------

    xjmin = x(jmin)
    yjmin = y(jmin)

    if (iszero(m1) && iszero(m2) && iszero(m3))        # All m's are zero
        wl[jmin] = 1.0
        wl[jmin+1] = 0.0
        jn = jmin + 1

    elseif iszero(yjmin)            # The second terms is either zero
        if iszero(xjmin)             # or undefined
            flag1 = 1
            jn = jmin
        else
            wl[jmin] = 1.0
            wl[jmin+1] = 0.0
            jn = jmin + 1
        end

    elseif (xjmin * yjmin >= 0.0) 
        # The second term is outside of the non-classical region
        wl[jmin] = 1.0
        wl[jmin+1] = -yjmin / xjmin
        jn = jmin + 1
    else
        # Calculate terms in the non-classical region
        rs[jmin] = -xjmin / yjmin

        jn = jmax

        for j = jmin+1:jmax-1
            denom = y(j) + z(j)*rs[j-1]
            xj = x(j)

            if (abs(xj) > abs(denom) || xj * denom >= 0 || iszero(denom)) 
                jn = j - 1
                break
            else
                rs[j] = -xj / denom
            end
        end

        wl[jn] = 1.0

        for k = 1:jn-jmin
            wl[jn-k] = wl[jn-k+1] * rs[jn-k]
        end

        if (jn == jmin) 
            # Calculate at least two terms so that these can be used
            # in three term recursion
            wl[jmin+1] = -yjmin / xjmin
            jn = jmin + 1
        end
    end

    if (jn == jmax) 
        # All terms are calculated
        @. w3j[jmin:jmax] = wl[jmin:jmax]
        normw3j!(w3j)
        fixsign!(w3j)
        return
    end

    #--------------------------------------------------------------------------
    #
    #   Calculate upper non-classical values for [jp, jmax].
    #   If the second last term can not be calculated because the
    #   recursion relations give a 1/0,  set flag2 to 1.
    #   (Note, I forn't think that this ever happens).
    #
    #--------------------------------------------------------------------------

    yjmax = y(jmax)
    zjmax = z(jmax)

    if (m1 == 0 && m2 == 0 && m3 == 0) 
        wu[jmax] = 1.0
        wu[jmax-1] = 0.0
        jp = jmax - 1

    elseif (yjmax == 0.0) 
        if (zjmax == 0.0) 
            flag2 = 1
            jp = jmax

        else
            wu[jmax] = 1.0
            wu[jmax-1] = - yjmax / zjmax
            jp = jmax-1

        end

    elseif (yjmax * zjmax >= 0.0) 
        wu[jmax] = 1.0
        wu[jmax-1] = - yjmax / zjmax
        jp = jmax - 1

    else
        rs[jmax] = -zjmax / yjmax

        jp = jmin

        for j=jmax-1:-1:jn
            denom = y(j) + x(j)*rs[j+1]
            zj = z(j)

            if (abs(zj) > abs(denom) || zj * denom >= 0.0 || iszero(denom)) 
                jp = j + 1
                break
            else
                rs[j] = -zj / denom
            end

        end

        wu[jp] = 1.0

        for k = 1:jmax-jp
            wu[jp+k] = wu[jp+k-1]*rs[jp+k]
        end

        if jp == jmax
            wu[jmax-1] = - yjmax / zjmax
            jp = jmax - 1
        end

    end

    #--------------------------------------------------------------------------
    #
    #   Calculate classical terms for [jn+1, jp-1] using standard three
    #   term rercusion relationship. Start from both jn and jp and stop at the
    #   midpoint. If flag1 is set,  perform the recursion solely from high
    #   to low values. If flag2 is set,  perform the recursion solely from
    #   low to high.
    #
    #--------------------------------------------------------------------------
    if (flag1 == 0) 
        jmid = div(jn + jp, 2)

        for j = jn:jmid - 1
            wl[j+1] = - (z(j)*wl[j-1] +y(j)*wl[j])/ x(j)

            if (abs(wl[j+1]) > 1) 
                # watch out for overflows.
                @. wl[jmin:j+1] /= scalef
            end

            if (abs(wl[j+1] / wl[j-1]) < 1 && !iszero(wl[j+1]))
                # If values are decreasing  stop upward iteration
                # and start with the forwnward iteration.
                jmid = j + 1
                break
            end

        end

        wnmid = wl[jmid]

        if (wl[jmid-1] != 0 && abs(wnmid / wl[jmid-1]) < 1e-6) 
            # Make sure that the stopping midpoint value is not a zero,
            # or close to it!
            wnmid = wl[jmid-1]
            jmid = jmid - 1
        end

        for j = jp:-1:jmid + 1
            wu[j-1] = - (x(j)*wu[j+1] + y(j)*wu[j] )/ z(j)
            if (abs(wu[j-1]) > 1) 
                @. wu[j-1:jmax] /= scalef
            end

        end

        wpmid = wu[jmid]

        # rescale two sequences to common midpoint
        if (jmid == jmax) 
            @. w3j[jmin:jmax] = wl[jmin:jmax]

        elseif (jmid == jmin) 
            @. w3j[jmin:jmax] = wu[jmin:jmax]

        else
            @. w3j[jmin:jmid] *= wpmid / wnmid
            @. w3j[jmid+1:jmax] = wu[jmid+1:jmax]
        end

    elseif (flag1 == 1 && flag2 == 0) 
        # iterature in forwnward direction only

        for j = jp:-1:jmin + 1
            wu[j-1] = - (x(j)*wu[j+1] + y(j)*wu[j] )/ z(j)

            if (abs(wu[j-1]) > 1) 
                @. wu[j-1:jmax] /= scalef
            end
        end

        @. w3j[jmin:jmax] = wu[jmin:jmax]

    elseif (flag2 == 1 && flag1 == 0) 
        # iterature in upward direction only

        for j = jn:jp-1
            wl[j+1] = - (z(j)*wl[j-1] +y(j)*wl[j])/ x(j)

            if (abs(wl[j+1]) > 1) 
                @. wl[jmin:j+1] /= scalef
            end

        end

        @. w3j[jmin:jmax] = wl[jmin:jmax]

    elseif (flag1 == 1 && flag2 == 1) 
        println("Error --- Wigner3j")
        println("Can not calculate function for input values, "*
                "both flag1 and flag 2 are set.")
        return

    end

    normw3j!(w3j);
    fixsign!(w3j);
    return
end

function wigner3j!(w3j::Vector{Float64},j2, j3, m1, m2, m3)
    jmin = max(abs(j2-j3), abs(m1)) :: Int
    wigner3j!(OffsetArray(w3j,jmin-1),j2, j3, m1, m2, m3)
end

function wigner3j(j2, j3, m1, m2, m3)
    jmin = max(abs(j2-j3), abs(m1)) :: Int
    jmax = (j2 + j3) :: Int
    w3j = zeros(jmin:jmax)
    wigner3j!(w3j,j2, j3, m1, m2, m3)
    return w3j
end

wigner3j(j2, j3, m2, m3) = wigner3j(j2, j3,-(m2+m3), m2, m3)
wigner3j!(w3j::AbstractArray,j2, j3, m2, m3) = wigner3j!(w3j,j2, j3,-(m2+m3), m2, m3)

