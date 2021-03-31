// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


//! \addtogroup fn_log_det
//! @{



//! log determinant of mat
template<typename T1>
inline
bool
log_det
  (
        typename T1::elem_type&          out_val,
        typename T1::pod_type&           out_sign,
  const Base<typename T1::elem_type,T1>& X,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = nullptr
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  Mat<eT> A(X.get_ref());
  
  arma_debug_check( (A.is_square() == false), "log_det(): given matrix must be square sized" );
  
  const bool status = auxlib::log_det(out_val, out_sign, A);
  
  if(status == false)
    {
    out_val  = eT(Datum<T>::nan);
    out_sign = T(0);
    
    arma_debug_warn_level(3, "log_det(): failed to find determinant");
    }
  
  return status;
  }



template<typename T1>
inline
bool
log_det
  (
        typename T1::elem_type& out_val,
        typename T1::pod_type&  out_sign,
  const Op<T1,op_diagmat>&      X,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = nullptr
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  const diagmat_proxy<T1> A(X.m);
  
  arma_debug_check( (A.n_rows != A.n_cols), "log_det(): given matrix must be square sized" );
  
  const uword N = (std::min)(A.n_rows, A.n_cols);
  
  if(N == 0)
    {
    out_val  = eT(0);
    out_sign =  T(1);
    
    return true;
    }
  
  eT x = A[0];
  
  T  sign = (is_cx<eT>::no) ?         ( (access::tmp_real(x) < T(0)) ?   T(-1) : T(1) ) : T(1);
  eT val  = (is_cx<eT>::no) ? std::log( (access::tmp_real(x) < T(0)) ? x*T(-1) : x    ) : std::log(x);
  
  for(uword i=1; i<N; ++i)
    {
    x = A[i];
    
    sign *= (is_cx<eT>::no) ?         ( (access::tmp_real(x) < T(0)) ?   T(-1) : T(1) ) : T(1);
    val  += (is_cx<eT>::no) ? std::log( (access::tmp_real(x) < T(0)) ? x*T(-1) : x    ) : std::log(x);
    }
  
  out_val  = val;
  out_sign = sign;
  
  return arma_isnan(out_val);
  }



template<typename T1>
inline
arma_warn_unused
std::complex<typename T1::pod_type>
log_det
  (
  const Base<typename T1::elem_type,T1>& X,
  const typename arma_blas_type_only<typename T1::elem_type>::result* junk = nullptr
  )
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  eT out_val  = eT(0);
   T out_sign =  T(0);
  
  Mat<eT> A(X.get_ref());
  
  arma_debug_check( (A.is_square() == false), "log_det(): given matrix must be square sized" );
  
  const bool status = auxlib::log_det(out_val, out_sign, A);
  
  if(status == false)
    {
    out_val  = eT(Datum<T>::nan);
    out_sign = T(0);
    
    arma_stop_runtime_error("log_det(): failed to find determinant");
    }
  
  return (out_sign >= T(1)) ? std::complex<T>(out_val) : (out_val + std::complex<T>(T(0),Datum<T>::pi));
  }



//! @}
