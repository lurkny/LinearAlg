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
  ( is_arma_type<T1>::value && is_arma_type<T2>::value && is_same_type<typename T1::elem_type, typename T2::elem_type>::value ),
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



// TODO:    mat = pow(mat.each_col(),    mat)  (no promotion)
// TODO:    mat = pow(mat.each_row(),    mat)  (no promotion)

// TODO: cx_mat = pow(cx_mat,            mat)  (promotion; ? implement via mtGlue to allow preservation of vector type info)

// TODO: cx_mat = pow(cx_mat.each_col(), mat)  (promotion)
// TODO: cx_mat = pow(cx_mat.each_row(), mat)  (promotion)



//! @}