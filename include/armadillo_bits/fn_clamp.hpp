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


//! \addtogroup fn_clamp
//! @{



template<typename T1>
arma_warn_unused
inline
typename
enable_if2
  <
  is_arma_type<T1>::value && is_cx<typename T1::elem_type>::no,
  const mtOp<typename T1::elem_type, T1, op_clamp>
  >::result
clamp(const T1& X, const typename T1::elem_type min_val, const typename T1::elem_type max_val)
  {
  arma_extra_debug_sigprint();
  
  return mtOp<typename T1::elem_type, T1, op_clamp>(mtOp_dual_aux_indicator(), X, min_val, max_val);
  }



template<typename T1>
arma_warn_unused
inline
typename
enable_if2
  <
  is_arma_type<T1>::value && is_cx<typename T1::elem_type>::yes,
  const mtOp<typename T1::elem_type, T1, op_clamp_cx>
  >::result
clamp(const T1& X, const typename T1::elem_type min_val, const typename T1::elem_type max_val)
  {
  arma_extra_debug_sigprint();
  
  return mtOp<typename T1::elem_type, T1, op_clamp_cx>(mtOp_dual_aux_indicator(), X, min_val, max_val);
  }



template<typename T1>
arma_warn_unused
inline
const mtOpCube<typename T1::elem_type, T1, op_clamp>
clamp(const BaseCube<typename T1::elem_type,T1>& X, const typename T1::elem_type min_val, const typename T1::elem_type max_val, typename arma_not_cx<typename T1::elem_type>::result* junk = nullptr)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return mtOpCube<typename T1::elem_type, T1, op_clamp>(mtOpCube_dual_aux_indicator(), X.get_ref(), min_val, max_val);
  }



template<typename T1>
arma_warn_unused
inline
const mtOpCube<typename T1::elem_type, T1, op_clamp_cx>
clamp(const BaseCube<typename T1::elem_type,T1>& X, const typename T1::elem_type min_val, const typename T1::elem_type max_val, typename arma_cx_only<typename T1::elem_type>::result* junk = nullptr)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return mtOpCube<typename T1::elem_type, T1, op_clamp_cx>(mtOpCube_dual_aux_indicator(), X.get_ref(), min_val, max_val);
  }



template<typename T1>
arma_warn_unused
inline
typename
enable_if2
  <
  is_cx<typename T1::elem_type>::no,
  SpMat<typename T1::elem_type>
  >::result
clamp(const SpBase<typename T1::elem_type,T1>& X, const typename T1::elem_type min_val, const typename T1::elem_type max_val)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  
  arma_debug_check( (min_val > max_val), "clamp(): min_val must be less than max_val" );
  
  SpMat<eT> out = X.get_ref();
  
  out.sync();
  
  const uword N = out.n_nonzero;
  
  eT* out_values = access::rwp(out.values);
  
  for(uword i=0; i<N; ++i)
    {
    eT& out_val = out_values[i];
    
    out_val = (out_val < min_val) ? min_val : ( (out_val > max_val) ? max_val : out_val );
    }
  
  if( (min_val == eT(0)) || (max_val == eT(0)) )  { out.remove_zeros(); }
  
  return out;
  }



template<typename T1>
arma_warn_unused
inline
typename
enable_if2
  <
  is_cx<typename T1::elem_type>::yes,
  SpMat<typename T1::elem_type>
  >::result
clamp(const SpBase<typename T1::elem_type,T1>& X, const typename T1::elem_type min_val, const typename T1::elem_type max_val)
  {
  arma_extra_debug_sigprint();
  
  typedef typename T1::elem_type eT;
  typedef typename T1::pod_type   T;
  
  const T min_val_real = std::real(min_val);
  const T min_val_imag = std::imag(min_val);
  
  const T max_val_real = std::real(max_val);
  const T max_val_imag = std::imag(max_val);
  
  arma_debug_check( (min_val_real > max_val_real), "clamp(): real(min_val) must be less than real(max_val)" );
  arma_debug_check( (min_val_imag > max_val_imag), "clamp(): imag(min_val) must be less than imag(max_val)" );
  
  SpMat<eT> out = X.get_ref();
  
  out.sync();
  
  const uword N = out.n_nonzero;
  
  eT* out_values = access::rwp(out.values);
  
  for(uword i=0; i<N; ++i)
    {
    eT& out_val = out_values[i];
    
    T val_real = std::real(out_val);
    T val_imag = std::imag(out_val);
    
    val_real = (val_real < min_val_real) ? min_val_real : ((val_real > max_val_real) ? max_val_real : val_real);
    val_imag = (val_imag < min_val_imag) ? min_val_imag : ((val_imag > max_val_imag) ? max_val_imag : val_imag);
    
    out_val = std::complex<T>(val_real,val_imag);
    }
  
  if( (min_val == eT(0)) || (max_val == eT(0)) )  { out.remove_zeros(); }
  
  return out;
  }



//! @}
