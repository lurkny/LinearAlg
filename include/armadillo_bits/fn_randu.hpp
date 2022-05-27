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


//! \addtogroup fn_randu
//! @{

// TODO: add variants with distr_param

arma_warn_unused
inline
double
randu()
  {
  return double(arma_rng::randu<double>());
  }



arma_warn_unused
inline
double
randu(const distr_param& param)
  {
  arma_extra_debug_sigprint();
  
  double a = double(0);
  double b = double(1);
  
  param.get_double_vals(a,b);
  
  arma_debug_check( (a >= b), "randu(): incorrect distribution parameters; a must be less than b" );
  
  double val = double(0);
  
  arma_rng::randu<double>::fill(&val, 1, a, b);
  
  return val;
  }



template<typename eT>
arma_warn_unused
inline
typename arma_real_or_cx_only<eT>::result
randu()
  {
  return eT(arma_rng::randu<eT>());
  }



template<typename eT>
arma_warn_unused
inline
typename arma_real_or_cx_only<eT>::result
randu(const distr_param& param)
  {
  arma_extra_debug_sigprint();
  
  double a = double(0);
  double b = double(1);
  
  param.get_double_vals(a,b);
  
  arma_debug_check( (a >= b), "randu(): incorrect distribution parameters; a must be less than b" );
  
  eT val = eT(0);
  
  arma_rng::randu<eT>::fill(&val, 1, a, b);
  
  return val;
  }



arma_warn_unused
arma_inline
const Gen<vec, gen_randu>
randu(const uword n_elem)
  {
  arma_extra_debug_sigprint();
  
  return Gen<vec, gen_randu>(n_elem, 1);
  }



arma_warn_unused
inline
vec
randu(const uword n_elem, const distr_param& param)
  {
  arma_extra_debug_sigprint();
  
  double a = double(0);
  double b = double(1);
  
  param.get_double_vals(a,b);
  
  arma_debug_check( (a >= b), "randu(): incorrect distribution parameters; a must be less than b" );
  
  vec out(n_elem, arma_nozeros_indicator());
  
  arma_rng::randu<double>::fill(out.memptr(), n_elem, a, b);
  
  return out;
  }



template<typename obj_type>
arma_warn_unused
arma_inline
const Gen<obj_type, gen_randu>
randu(const uword n_elem, const arma_empty_class junk1 = arma_empty_class(), const typename arma_Mat_Col_Row_only<obj_type>::result* junk2 = nullptr)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk1);
  arma_ignore(junk2);
  
  const uword n_rows = (is_Row<obj_type>::value) ? uword(1) : n_elem;
  const uword n_cols = (is_Row<obj_type>::value) ? n_elem   : uword(1);
  
  return Gen<obj_type, gen_randu>(n_rows, n_cols);
  }



arma_warn_unused
arma_inline
const Gen<mat, gen_randu>
randu(const uword n_rows, const uword n_cols)
  {
  arma_extra_debug_sigprint();
  
  return Gen<mat, gen_randu>(n_rows, n_cols);
  }



arma_warn_unused
arma_inline
const Gen<mat, gen_randu>
randu(const SizeMat& s)
  {
  arma_extra_debug_sigprint();
  
  return Gen<mat, gen_randu>(s.n_rows, s.n_cols);
  }



template<typename obj_type>
arma_warn_unused
arma_inline
const Gen<obj_type, gen_randu>
randu(const uword n_rows, const uword n_cols, const typename arma_Mat_Col_Row_only<obj_type>::result* junk = nullptr)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  if(is_Col<obj_type>::value)
    {
    arma_debug_check( (n_cols != 1), "randu(): incompatible size" );
    }
  else
  if(is_Row<obj_type>::value)
    {
    arma_debug_check( (n_rows != 1), "randu(): incompatible size" );
    }
  
  return Gen<obj_type, gen_randu>(n_rows, n_cols);
  }



template<typename obj_type>
arma_warn_unused
arma_inline
const Gen<obj_type, gen_randu>
randu(const SizeMat& s, const typename arma_Mat_Col_Row_only<obj_type>::result* junk = nullptr)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return randu<obj_type>(s.n_rows, s.n_cols);
  }



arma_warn_unused
arma_inline
const GenCube<cube::elem_type, gen_randu>
randu(const uword n_rows, const uword n_cols, const uword n_slices)
  {
  arma_extra_debug_sigprint();
  
  return GenCube<cube::elem_type, gen_randu>(n_rows, n_cols, n_slices);
  }



arma_warn_unused
arma_inline
const GenCube<cube::elem_type, gen_randu>
randu(const SizeCube& s)
  {
  arma_extra_debug_sigprint();
  
  return GenCube<cube::elem_type, gen_randu>(s.n_rows, s.n_cols, s.n_slices);
  }



template<typename cube_type>
arma_warn_unused
arma_inline
const GenCube<typename cube_type::elem_type, gen_randu>
randu(const uword n_rows, const uword n_cols, const uword n_slices, const typename arma_Cube_only<cube_type>::result* junk = nullptr)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return GenCube<typename cube_type::elem_type, gen_randu>(n_rows, n_cols, n_slices);
  }



template<typename cube_type>
arma_warn_unused
arma_inline
const GenCube<typename cube_type::elem_type, gen_randu>
randu(const SizeCube& s, const typename arma_Cube_only<cube_type>::result* junk = nullptr)
  {
  arma_extra_debug_sigprint();
  arma_ignore(junk);
  
  return GenCube<typename cube_type::elem_type, gen_randu>(s.n_rows, s.n_cols, s.n_slices);
  }



//! @}
