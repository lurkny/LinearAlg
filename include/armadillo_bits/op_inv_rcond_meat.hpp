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


//! \addtogroup op_inv_rcond
//! @{



template<typename T1>
inline
bool
op_inv_rcond::apply_direct_gen(Mat<typename T1::elem_type>& out, typename T1::pod_type& out_rcond, const Base<typename T1::elem_type,T1>& expr)
  {
  arma_extra_debug_sigprint();
  
  // NOTE: this is a work in progress
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  out       = expr.get_ref();
  out_rcond = T(0);
  
  arma_debug_check( (out.is_square() == false), "inv(): given matrix must be square sized" );
  
  const uword N = out.n_rows;
  
  if(is_op_diagmat<T1>::value || out.is_diagmat())
    {
    arma_extra_debug_print("op_inv_rcond: detected diagonal matrix");
    
    eT* colmem = out.memptr();
    
    T max_abs_src_val = T(0);
    T max_abs_inv_val = T(0);
    
    for(uword i=0; i<N; ++i)
      {
      eT& out_ii = colmem[i];
      
      const eT src_val = out_ii;
      const eT inv_val = eT(1) / src_val;
      
      if(src_val == eT(0))  { return false; }
      
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
  
  const strip_trimat<T1> strip(expr.get_ref());
  
  const bool is_triu_expr = strip.do_triu;
  const bool is_tril_expr = strip.do_tril;
  
  const bool is_triu_mat = (is_triu_expr || is_tril_expr) ? false : (                        trimat_helper::is_triu(out));
  const bool is_tril_mat = (is_triu_expr || is_tril_expr) ? false : ((is_triu_mat) ? false : trimat_helper::is_tril(out));
  
  if(is_triu_expr || is_tril_expr || is_triu_mat || is_tril_mat)
    {
    return auxlib::inv_tr_rcond(out, out_rcond, ((is_triu_expr || is_triu_mat) ? uword(0) : uword(1)));
    }
  
  return auxlib::inv_rcond(out, out_rcond);
  }



template<typename T1>
inline
bool
op_inv_rcond::apply_direct_spd(Mat<typename T1::elem_type>& out, typename T1::pod_type& out_rcond, const Base<typename T1::elem_type,T1>& expr)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  out       = expr.get_ref();
  out_rcond = T(0);
  
  arma_debug_check( (out.is_square() == false), "inv_sympd(): given matrix must be square sized" );
  
  if((arma_config::debug) && (auxlib::rudimentary_sym_check(out) == false))
    {
    if(is_cx<eT>::no )  { arma_debug_warn_level(1, "inv_sympd(): given matrix is not symmetric"); }
    if(is_cx<eT>::yes)  { arma_debug_warn_level(1, "inv_sympd(): given matrix is not hermitian"); }
    }
  
  const uword N = out.n_rows;
  
  if(is_cx<eT>::yes)
    {
    arma_extra_debug_print("op_inv_rcond: checking imaginary components of diagonal elements");
    
    const T tol = T(100) * std::numeric_limits<T>::epsilon();  // allow some leeway
    
    const eT* colmem = out.memptr();
    
    for(uword i=0; i<N; ++i)
      {
      const eT& out_ii      = colmem[i];
      const  T  out_ii_imag = access::tmp_imag(out_ii);
      
      if(std::abs(out_ii_imag) > tol)  { return false; }
      
      colmem += N;
      }
    }
  
  if(is_op_diagmat<T1>::value || out.is_diagmat())
    {
    arma_extra_debug_print("op_inv_rcond: detected diagonal matrix");
    
    eT* colmem = out.memptr();
    
    T max_abs_src_val = T(0);
    T max_abs_inv_val = T(0);
    
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
  
  return auxlib::inv_sympd_rcond(out, out_rcond, T(-1));
  }



//! @}
