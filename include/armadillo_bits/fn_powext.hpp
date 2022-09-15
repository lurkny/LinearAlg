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


//! \addtogroup fn_powext
//! @{



template<typename T1, typename T2>
arma_warn_unused
arma_inline
typename
enable_if2
  <
  ( is_arma_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::yes ),
  const Glue<T1, T2, glue_powext>
  >::result
pow
  (
  const T1& X,
  const T2& Y
  )
  {
  arma_extra_debug_sigprint();
  
  return Glue<T1, T2, glue_powext>(X, Y);
  }



template<typename T1, typename T2>
arma_warn_unused
arma_inline
typename
enable_if2
  <
  ( is_arma_type<T1>::value && is_arma_type<T2>::value && is_cx<typename T1::elem_type>::yes && is_same_type<typename T1::pod_type, typename T2::elem_type>::yes ),
  const mtGlue<typename T1::elem_type, T1, T2, glue_powext_cx>
  >::result
pow
  (
  const T1& X,
  const T2& Y
  )
  {
  arma_extra_debug_sigprint();
  
  return mtGlue<typename T1::elem_type, T1, T2, glue_powext_cx>(X, Y);
  }



template<typename parent, unsigned int mode, typename T2>
arma_warn_unused
inline
Mat<typename parent::elem_type>
pow
  (
  const subview_each1<parent,mode>&          X,
  const Base<typename parent::elem_type,T2>& Y
  )
  {
  arma_extra_debug_sigprint();
  
  return glue_powext::apply(X,Y);
  }



template<typename parent, unsigned int mode, typename T2>
arma_warn_unused
inline
typename
enable_if2
  <
  ( is_cx<typename parent::elem_type>::yes && is_same_type<typename parent::pod_type, typename T2::elem_type>::yes ),
  Mat<typename parent::elem_type>
  >::result
pow
  (
  const subview_each1<parent,mode>&      X,
  const Base<typename T2::elem_type,T2>& Y
  )
  {
  arma_extra_debug_sigprint();
  
  return glue_powext_cx::apply(X,Y);
  }



//! @}
