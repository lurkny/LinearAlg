// SPDX-License-Identifier: Apache-2.0
// 
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


//! \addtogroup op_inv_spd
//! @{



template<typename T1>
inline
void
op_inv_spd_default::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_inv_spd_default>& X)
  {
  arma_extra_debug_sigprint();
  
  const bool status = op_inv_spd_default::apply_direct(out, X.m);
  
  if(status == false)
    {
    out.soft_reset();
    arma_stop_runtime_error("inv_sympd(): matrix is singular or not positive definite");
    }
  }



template<typename T1>
inline
bool
op_inv_spd_default::apply_direct(Mat<typename T1::elem_type>& out, const Base<typename T1::elem_type,T1>& expr)
  {
  arma_extra_debug_sigprint();
  
  return op_inv_spd_full::apply_direct<T1,false>(out, expr, uword(0));
  }



//



template<typename T1>
inline
void
op_inv_spd_full::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_inv_spd_full>& X)
  {
  arma_extra_debug_sigprint();
  
  const uword flags = X.in_aux_uword_a;
  
  const bool status = op_inv_spd_full::apply_direct(out, X.m, flags);
  
  if(status == false)
    {
    out.soft_reset();
    arma_stop_runtime_error("inv_sympd(): matrix is singular or not positive definite");
    }
  }



template<typename T1, const bool has_user_flags>
inline
bool
op_inv_spd_full::apply_direct(Mat<typename T1::elem_type>& out, const Base<typename T1::elem_type,T1>& expr, const uword flags)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  if(has_user_flags == true )  { arma_extra_debug_print("op_inv_spd_full: has_user_flags = true");  }
  if(has_user_flags == false)  { arma_extra_debug_print("op_inv_spd_full: has_user_flags = false"); }
  
  const bool tiny         = has_user_flags && bool(flags & inv_opts::flag_tiny        );
  const bool likely_sympd = has_user_flags && bool(flags & inv_opts::flag_likely_sympd);
  const bool no_sympd     = has_user_flags && bool(flags & inv_opts::flag_no_sympd    );
  
  arma_extra_debug_print("op_inv_spd_full: enabled flags:");
  
  if(tiny        )  { arma_extra_debug_print("tiny");         }
  if(likely_sympd)  { arma_extra_debug_print("likely_sympd"); }
  if(no_sympd    )  { arma_extra_debug_print("no_sympd");     }
  
  if(likely_sympd)  { arma_debug_warn_level(1, "inv_sympd(): option 'likely_sympd' ignored" ); }
  if(no_sympd)      { arma_debug_warn_level(1, "inv_sympd(): option 'no_sympd' ignored" );     }
  
  out = expr.get_ref();
  
  arma_debug_check( (out.is_square() == false), "inv_sympd(): given matrix must be square sized" );
  
  if((arma_config::debug) && (arma_config::warn_level > 0))
    {
    if(auxlib::rudimentary_sym_check(out) == false)
      {
      if(is_cx<eT>::no )  { arma_debug_warn_level(1, "inv_sympd(): given matrix is not symmetric"); }
      if(is_cx<eT>::yes)  { arma_debug_warn_level(1, "inv_sympd(): given matrix is not hermitian"); }
      }
    else
    if((is_cx<eT>::yes) && (sympd_helper::check_diag_imag(out) == false))
      {
      arma_debug_warn_level(1, "inv_sympd(): imaginary components on diagonal are non-zero");
      }
    }
  
  const uword N = out.n_rows;
  
  if(is_cx<eT>::no)
    {
    if(N == 1)
      {
      const T a = access::tmp_real(out[0]);
      
      out[0] = eT(T(1) / a);
      
      return (a > T(0));
      }
    else
    if(N == 2)
      {
      const bool status = op_inv_spd_full::apply_tiny_2x2(out);
      
      if(status)  { return true; }
      }
    
    // NOTE: currently the "tiny" option is not used,
    // NOTE: as it's cumbersome to implement optimisations for 3x3 and 4x4 matrices;
    // NOTE: need to ensure that all leading principal minors are positive,
    // NOTE: which would involve a lot of dedicated code
    
    // fallthrough if optimisation failed
    }
  
  if(is_op_diagmat<T1>::value || out.is_diagmat())
    {
    arma_extra_debug_print("op_inv_spd_full: detected diagonal matrix");
    
    eT* colmem = out.memptr();
    
    for(uword i=0; i<N; ++i)
      {
            eT& out_ii      = colmem[i];
      const  T  out_ii_real = access::tmp_real(out_ii);
      
      if(out_ii_real <= T(0))  { return false; }
      
      out_ii = eT(T(1) / out_ii_real);
      
      colmem += N;
      }
    
    return true;
    }
  
  bool sympd_state_junk = false;
  
  return auxlib::inv_sympd(out, sympd_state_junk);
  }



template<typename eT>
arma_cold
inline
bool
op_inv_spd_full::apply_tiny_2x2(Mat<eT>& X)
  {
  arma_extra_debug_sigprint();
  
  typedef typename get_pod_type<eT>::result T;
  
  // NOTE: assuming matrix X is square sized
  // NOTE: assuming matrix X is symmetric
  // NOTE: assuming matrix X is real
  
  constexpr T det_min =        std::numeric_limits<T>::epsilon();
  constexpr T det_max = T(1) / std::numeric_limits<T>::epsilon();
  
  eT* Xm = X.memptr();
  
  const T a = access::tmp_real(Xm[pos<0,0>::n2]);
  const T c = access::tmp_real(Xm[pos<1,0>::n2]);
  const T d = access::tmp_real(Xm[pos<1,1>::n2]);
  
  const T det_val = (a*d - c*c);
  
  // positive definite iff all leading principal minors are positive
  // a       = first  leading principal minor (top-left 1x1 submatrix)
  // det_val = second leading principal minor (top-left 2x2 submatrix)
  
  if(a <= T(0))  { return false; }
  
  // NOTE: since det_min is positive, this also checks whether det_val is positive
  if((det_val < det_min) || (det_val > det_max))  { return false; }
  
  Xm[pos<0,0>::n2] =  d / det_val;
  Xm[pos<0,1>::n2] = -c / det_val;
  Xm[pos<1,0>::n2] = -c / det_val;
  Xm[pos<1,1>::n2] =  a / det_val;
  
  return true;
  }



//



template<typename T1>
inline
bool
op_inv_spd_rcond::apply_direct(Mat<typename T1::elem_type>& out, typename T1::pod_type& out_rcond, const Base<typename T1::elem_type,T1>& expr)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  out       = expr.get_ref();
  out_rcond = T(0);
  
  arma_debug_check( (out.is_square() == false), "inv_sympd(): given matrix must be square sized" );
  
  if((arma_config::debug) && (arma_config::warn_level > 0))
    {
    if(auxlib::rudimentary_sym_check(out) == false)
      {
      if(is_cx<eT>::no )  { arma_debug_warn_level(1, "inv_sympd(): given matrix is not symmetric"); }
      if(is_cx<eT>::yes)  { arma_debug_warn_level(1, "inv_sympd(): given matrix is not hermitian"); }
      }
    else
    if((is_cx<eT>::yes) && (sympd_helper::check_diag_imag(out) == false))
      {
      arma_debug_warn_level(1, "inv_sympd(): imaginary components on diagonal are non-zero");
      }
    }
  
  if(is_op_diagmat<T1>::value || out.is_diagmat())
    {
    arma_extra_debug_print("op_inv_spd_rcond: detected diagonal matrix");
    
    eT* colmem = out.memptr();
    
    T max_abs_src_val = T(0);
    T max_abs_inv_val = T(0);
    
    const uword N = out.n_rows;
    
    for(uword i=0; i<N; ++i)
      {
      eT& out_ii = colmem[i];
      
      const eT src_val = out_ii;
      const eT inv_val = eT(1) / src_val;
      
      if( (src_val == eT(0)) || (access::tmp_real(src_val) <= T(0)) )  { return false; }
      
      out_ii = inv_val;
      
      const T abs_src_val = std::abs(src_val);
      const T abs_inv_val = std::abs(inv_val);
      
      max_abs_src_val = (abs_src_val > max_abs_src_val) ? abs_src_val : max_abs_src_val;
      max_abs_inv_val = (abs_inv_val > max_abs_inv_val) ? abs_inv_val : max_abs_inv_val;
      
      colmem += N;
      }
    
    out_rcond = T(1) / (max_abs_src_val * max_abs_inv_val);
    
    return true;
    }
  
  if(auxlib::crippled_lapack(out))
    {
    arma_extra_debug_print("op_inv_spd_rcond: workaround for crippled lapack");
    
    Mat<eT> tmp = out;
    
    bool sympd_state = false;
    
    auxlib::inv_sympd(out, sympd_state);
    
    if(sympd_state == false)  { out.soft_reset(); out_rcond = T(0); return false; }
    
    out_rcond = auxlib::rcond(tmp);
    
    if(out_rcond == T(0))  { out.soft_reset(); return false; }
    
    return true;
    }
  
  bool is_sympd_junk = false;
  
  return auxlib::inv_sympd_rcond(out, is_sympd_junk, out_rcond, T(-1));
  }



//! @}
