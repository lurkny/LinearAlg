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
  
  return op_inv_spd::apply_direct<T1,false>(out, expr, uword(0));
  }



//



template<typename T1>
inline
void
op_inv_spd::apply(Mat<typename T1::elem_type>& out, const Op<T1,op_inv_spd>& X)
  {
  arma_extra_debug_sigprint();
  
  const uword flags = X.in_aux_uword_a;
  
  const bool status = op_inv_spd::apply_direct(out, X.m, flags);
  
  if(status == false)
    {
    out.soft_reset();
    arma_stop_runtime_error("inv_sympd(): matrix is singular or not positive definite");
    }
  }



template<typename T1, const bool has_user_flags>
inline
bool
op_inv_spd::apply_direct(Mat<typename T1::elem_type>& out, const Base<typename T1::elem_type,T1>& expr, const uword flags)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  if(has_user_flags == true )  { arma_extra_debug_print("op_inv_spd: has_user_flags == true");  }
  if(has_user_flags == false)  { arma_extra_debug_print("op_inv_spd: has_user_flags == false"); }
  
  const bool fast         = has_user_flags && bool(flags & inv_opts::flag_fast        );
  const bool likely_sympd = has_user_flags && bool(flags & inv_opts::flag_likely_sympd);
  const bool no_sympd     = has_user_flags && bool(flags & inv_opts::flag_no_sympd    );
  
  arma_extra_debug_print("op_inv_gen: enabled flags:");
  
  if(fast        )  { arma_extra_debug_print("fast");         }
  if(likely_sympd)  { arma_extra_debug_print("likely_sympd"); }
  if(no_sympd    )  { arma_extra_debug_print("no_sympd");     }
  
  if(likely_sympd)  { arma_debug_warn_level(1, "inv_sympd(): option 'likely_sympd' ignored" ); }
  if(no_sympd)      { arma_debug_warn_level(1, "inv_sympd(): option 'no_sympd' ignored" );     }
  
  out = expr.get_ref();
  
  arma_debug_check( (out.is_square() == false), "inv_sympd(): given matrix must be square sized" );
  
  if((arma_config::debug) && (auxlib::rudimentary_sym_check(out) == false))
    {
    if(is_cx<eT>::no )  { arma_debug_warn_level(1, "inv_sympd(): given matrix is not symmetric"); }
    if(is_cx<eT>::yes)  { arma_debug_warn_level(1, "inv_sympd(): given matrix is not hermitian"); }
    }
  
  const uword N = (std::min)(out.n_rows, out.n_cols);
  
  if((is_cx<eT>::no) && (is_op_diagmat<T1>::value || out.is_diagmat()))
    {
    arma_extra_debug_print("op_inv_spd: detected diagonal matrix");
    
    // specialised handling of real matrices only;
    // currently auxlib::inv_sympd() does not enforce that 
    // imaginary components of diagonal elements must be zero;
    // strictly enforcing this constraint may break existing user software.
    
    for(uword i=0; i<N; ++i)
      {
            eT&      out_ii = out.at(i,i);
      const  T  real_out_ii = access::tmp_real(out_ii);
      
      if(real_out_ii <= T(0))  { return false; }
      
      out_ii = eT(T(1) / real_out_ii);
      }
      
    return true;
    }
  
  // TODO: the tinymatrix optimisation currently does not care if the given matrix is not sympd;
  // TODO: need to print a warning if the matrix is not sympd based on fast rudimentary checks,
  // TODO: ie. diagonal values are > 0, and max value is on the diagonal.
  // 
  // TODO: when the major version is bumped:
  // TODO: either rework the tinymatrix optimisation to be reliably more strict, or remove it entirely
  
  if((is_cx<eT>::no) && (N <= 4) && (fast))
    {
    arma_extra_debug_print("op_inv_spd: attempting tinymatrix optimisation");
    
    T max_diag = T(0);
    
    const eT* colmem = out.memptr();
    
    for(uword i=0; i<N; ++i)
      {
      const eT&      out_ii = colmem[i];
      const  T  real_out_ii = access::tmp_real(out_ii);
      
      if(real_out_ii <= T(0))  { return false; }
      
      max_diag = (real_out_ii > max_diag) ? real_out_ii : max_diag;
      
      colmem += N;
      }
    
    colmem = out.memptr();
    
    for(uword c=0; c < N; ++c)
      {
      for(uword r=c; r < N; ++r)
        {
        const T abs_val = std::abs(colmem[r]);
        
        if(abs_val > max_diag)  { return false; }
        }
      
      colmem += N;
      }
    
    Mat<eT> tmp(out.n_rows, out.n_rows, arma_nozeros_indicator());
    
    const bool status = op_inv_gen::apply_tiny_noalias(tmp, out);
    
    if(status)  { arrayops::copy(out.memptr(), tmp.memptr(), tmp.n_elem); return true; }
    
    arma_extra_debug_print("op_inv_spd: tinymatrix optimisation failed");
    
    // fallthrough if optimisation failed
    }
  
  return auxlib::inv_sympd(out);
  }



//! @}
