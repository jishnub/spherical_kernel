!************************************************************************************************************************************************************ 
!  FUNCTIONS: Wigner's d-matrix
!  Methods: (a) Complex Fourier-series expansion of the d-matrix; 
!           (b) the Fourier coefficients are obtained by numerical diagonalizing the angular-momentum operator Jy, using the ZHBEV subroutine of LAPACK.
!
!  update:  (y/m/d) 2015/09/30 by G.R.J
!*************************************************************************************************************************************************************
!    
!   [1] The defination of the term X_{m}=<j,m-1|J_{-}|j,m>, where J_{-} is the ladder operator, 
!       obeying J_{-}|j,m>=X_{m} |j,m-1>, and {|j,m>} eigenvectors of the angular-momentum operator Jz.  
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
    function X(j,n)
  real*8 X,n,j
    X=dsqrt((j+n)*(j-n+1.d0))
  end 

!****************************************************************************************************
!   [2]  The Hermitian matrix of J_{y}
!******************************************************************************************************
  subroutine coeffi(A,Ndim)
  implicit real*8 (a-h,o-z)
  external X
  integer, intent(in) :: Ndim
  integer m
    real*8 jmq
    complex*16 im
  COMPLEX*16, intent(out) :: A(Ndim,Ndim)
! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    im=(0.d0,1.d0)
    jmq=(dble(Ndim)-1.d0)/2.d0
    do i=1,Ndim
    do j=1,Ndim
            A(i,j)=0.d0

           if(i.eq.1) then
          A(i,i+1)=X(jmq,dble(i)-jmq)*im/(2.d0)
            endif
            if(i.eq.Ndim) then
          A(i,i-1)=-X(jmq,-dble(i)+jmq+2)*im/(2.d0)
            endif
             
         if((i.gt.1).and.(i.lt.Ndim))then
              A(i,i+1)=X(jmq,dble(i)-jmq)*im/(2.d0)
              A(i,i-1)=-X(jmq,-dble(i)+jmq+2)*im/(2.d0)
         endif
        enddo
    enddo
  return
    end

!**************************************************************************************************************************
!   [3] Calculation of the eigenvalues and the right eigenvectors for the Hermitian matrix J_{y}, 
!       using ZHBEV subroutine of LAPACK.            
!****************************************************************************************************************************
    subroutine eigen(A,Eigenvalue,Eigenvec,Ndim)
    character *1 JOBZ,UPLO
    integer, intent(in) :: Ndim
    integer KD,LDAB,LDZ,INFO
    double precision, intent(out) :: Eigenvalue(Ndim)
    double precision,allocatable:: RWORK(:)
    double complex,allocatable:: WORK(:),AB(:,:)
    double complex, intent(in) :: A(Ndim,Ndim)
    double complex, intent(out) :: Eigenvec(Ndim,Ndim)
! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   JOBZ='V'
   UPLO='U'
   KD=1
   LDAB=KD+1
   LDZ=max(1,Ndim)
   INFO=0
   allocate(AB(LDAB,Ndim),WORK(Ndim),RWORK(max(1,3*Ndim-2)))

   ! call coeffi(A,Ndim)
      do i=1,Ndim
      do j=i,Ndim
       if(max(1,j-kd).le.i) then
            AB(KD+1+i-j,j)=A(i,j)
       end if
          enddo
        enddo

   call ZHBEV (JOBZ, UPLO, Ndim, KD, AB, LDAB, Eigenvalue, Eigenvec, LDZ, WORK, RWORK,INFO)

   deallocate(RWORK,WORK,AB)

   end


!**************************************************************************************************************************
!   [4] The subroutine to calculate the Wigner-d matrix and its first-order derivative with various mv and nv for a given {j,theta}
!****************************************************************************************************************************
  subroutine Wigner_dmatrix(jmq,Ndim,theta,wd_matrix,diffwd_matrix)
  integer ixx,iyy,mu
  real*8, intent(in) :: jmq, theta
  integer, intent(in) :: Ndim
    real *8 mvar, nvar, wd, wdderivative
    real*8, intent(out) :: wd_matrix(Ndim,Ndim), diffwd_matrix(Ndim,Ndim)
  double complex  inum
  double precision,allocatable :: Evalue(:),Eigenvalue(:)
    double complex,allocatable :: coeffimatrix(:,:), Eigenvector(:,:)
! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
  inum=(0.d0,1.d0)
    ! Ndim=int(2.d0*jmq+1.d0)

    allocate(Evalue(Ndim),coeffimatrix(Ndim,Ndim),Eigenvector(Ndim,Ndim),Eigenvalue(Ndim))

    call coeffi(coeffimatrix,Ndim)   
    call eigen(coeffimatrix,Eigenvalue,Eigenvector,Ndim)

     do ixx=1,Ndim
     do iyy=1,Ndim
         
        wd=0.d0
      wdderivative=0.d0
     
        do mu=1,Ndim
         if(dble(int(jmq)).eq.jmq) then
                Evalue(mu)=dble(floor(Real(Eigenvalue(mu))+0.5d0))
       else
                Evalue(mu)=dble(floor(Real(Eigenvalue(mu)))+0.5d0)
       endif
                wd=wd+exp(-inum*Evalue(mu)*theta)*Eigenvector(ixx,mu)*dconjg(Eigenvector(iyy,mu))
                wdderivative=wdderivative+(-inum*Evalue(mu))*exp(-inum*Evalue(mu)*theta)*&
                Eigenvector(ixx,mu)*dconjg(Eigenvector(iyy,mu))
      enddo    
      
      wd_matrix(ixx,iyy)=wd
      diffwd_matrix(ixx,iyy)=wdderivative
    end do
   end do
                            
    deallocate(Evalue, coeffimatrix, Eigenvector,Eigenvalue)
  return
    end


!*********************************************************************************************************************
! [5] MAIN PROGRAM
!*********************************************************************************************************************
!   Program Main
!   external Wigner_dmatrix 
!   real*8 jmq,mv,nv,theta,pi
!   integer Ndim
!   double precision,allocatable :: wd_matrix(:,:), diffwd_matrix(:,:)

!     pi=acos(-1d0)

! !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! !   (i) Compute an element of the Wigner's d-matrix, i.e., d^{j}_{mv,nv}(\theta) for given {j,mv,nv,theta}, with the results printed on screen.
! ! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!    if(1.eq.0) then   
!     jmq=1/2.d0
!   Ndim=int(2.d0*jmq+1.d0)
!   mv=1/2.d0
!   nv=-1/2.d0
!   theta=pi/dble(6.0)
            
!         allocate(wd_matrix(Ndim,Ndim), diffwd_matrix(Ndim,Ndim))

!       call Wigner_dmatrix(jmq,theta,wd_matrix,diffwd_matrix)
                          
!     print *, "For given {j,m,n,\theta}=", jmq, ",", mv, ",", nv, ",", theta, ", the element of Wigner's d-matrix=", wd_matrix(jmq+mv+1,jmq+nv+1), ", and its 1th-order derivative=", diffwd_matrix(jmq+mv+1,jmq+nv+1)
             
!       deallocate(wd_matrix, diffwd_matrix)
!     pause
!    endif
! !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! !   (ii) Compute all the elements of the Wigner's d-matrix for given {j,theta}. 
! ! ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!    open(1,FILE='wdj100_18piov36.dat',status='unknown')
!    open(2,FILE='wdj100diff_18piov36.dat',status='unknown')

!      jmq=100.d0
!    Ndim=int(2.d0*jmq+1.d0)
!      theta=18.d0*pi/dble(36.0)
            
!      allocate(wd_matrix(Ndim,Ndim),diffwd_matrix(Ndim,Ndim))
!     call Wigner_dmatrix(jmq,theta,wd_matrix,diffwd_matrix)
!     do i=1,Ndim
!             mv=-dble(Ndim-1)/2.0+dble(i)-1.d0
!           do j=1,Ndim
!                 nv=-dble(Ndim-1)/2.0+dble(j)-1.d0
!                write(1,*) mv, nv, wd_matrix(i,j)
!            write(2,*) mv, nv, diffwd_matrix(i,j)
!              print *, mv,nv, wd_matrix(i,j) 
!             end do 
!      end do  
!    deallocate(wd_matrix, diffwd_matrix)
!    end
