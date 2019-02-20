module shtools_wrapper

use, intrinsic :: iso_c_binding
use SHTOOLS
implicit none

contains 

subroutine wigner3j_wrapper(w3j,len,j2, j3, m1, m2, m3, exitstatus) bind(C, name="wigner3j_wrapper")
    integer(kind=C_INT32_T), intent(in)  :: len,j2,j3,m1,m2,m3
    real(kind=C_DOUBLE), intent(out) :: w3j(len)
    integer(kind=C_INT32_T) :: jmin,jmax
    integer(kind=C_INT32_T), intent(out), optional :: exitstatus

    call Wigner3j(w3j, jmin, jmax, j2, j3, m1, m2, m3, exitstatus=exitstatus)
end subroutine wigner3j_wrapper

end module shtools_wrapper

program main
    use, intrinsic :: iso_c_binding
    use shtools_wrapper
    implicit none
    integer(kind=C_INT32_T) len,jmin, jmax, j2, j3, m1, m2, m3, exitstatus
    real(kind=C_DOUBLE), allocatable :: w3j(:)

    j2=1; j3=1; m1=0; m2=1; m3=-1

    exitstatus = 0
    len = j2+j3+1
    allocate(w3j(len))
    w3j = 0.

    call wigner3j_wrapper(w3j,len,j2, j3, m1, m2, m3, exitstatus)

    print*,w3j

    deallocate(w3j)

end program main