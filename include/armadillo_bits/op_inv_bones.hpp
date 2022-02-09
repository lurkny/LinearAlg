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


//! \addtogroup op_inv
//! @{



class op_inv_gen_default
  : public traits_op_default
  {
  public:
  
  template<typename T1>
  inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_inv_gen_default>& in);
  
  template<typename T1>
  inline static bool apply_direct(Mat<typename T1::elem_type>& out, const Base<typename T1::elem_type,T1>& expr, const char* caller_sig);
  };



class op_inv_gen
  : public traits_op_default
  {
  public:
  
  template<const uword row, const uword col>
  struct pos
    {
    static constexpr uword n2 = row + col*2;
    static constexpr uword n3 = row + col*3;
    static constexpr uword n4 = row + col*4;
    };
  
  template<typename T1>
  inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_inv_gen>& in);
  
  template<typename T1, const bool has_user_flags = true>
  inline static bool apply_direct(Mat<typename T1::elem_type>& out, const Base<typename T1::elem_type,T1>& expr, const char* caller_sig, const uword flags);
  
  template<typename T1>
  inline static bool apply_diagmat(Mat<typename T1::elem_type>& out, const T1& X, const char* caller_sig);
  
  template<typename eT>
  arma_cold inline static bool apply_tiny_noalias(Mat<eT>& out, const Mat<eT>& X);
  };



class op_inv_spd_default
  : public traits_op_default
  {
  public:
  
  template<typename T1>
  inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1,op_inv_spd_default>& in);
  
  template<typename T1>
  inline static bool apply_direct(Mat<typename T1::elem_type>& out, const Base<typename T1::elem_type,T1>& expr);
  };



namespace inv_opts
  {
  struct opts
    {
    const uword flags;
    
    inline explicit opts(const uword in_flags);
    
    inline const opts operator+(const opts& rhs) const;
    };
  
  inline
  opts::opts(const uword in_flags)
    : flags(in_flags)
    {}
  
  inline
  const opts
  opts::operator+(const opts& rhs) const
    {
    const opts result( flags | rhs.flags );
    
    return result;
    }
  
  // The values below (eg. 1u << 1) are for internal Armadillo use only.
  // The values can change without notice.
  
  static constexpr uword flag_none         = uword(0       );
  static constexpr uword flag_fast         = uword(1u <<  0);
  static constexpr uword flag_triu         = uword(1u <<  1);
  static constexpr uword flag_tril         = uword(1u <<  2);
  static constexpr uword flag_likely_sympd = uword(1u <<  3);
  static constexpr uword flag_no_trimat    = uword(1u <<  4);
  static constexpr uword flag_no_sympd     = uword(1u <<  5);
  
  struct opts_none         : public opts { inline opts_none()         : opts(flag_none        ) {} };
  struct opts_fast         : public opts { inline opts_fast()         : opts(flag_fast        ) {} };
  struct opts_triu         : public opts { inline opts_triu()         : opts(flag_triu        ) {} };
  struct opts_tril         : public opts { inline opts_tril()         : opts(flag_tril        ) {} };
  struct opts_likely_sympd : public opts { inline opts_likely_sympd() : opts(flag_likely_sympd) {} };
  struct opts_no_trimat    : public opts { inline opts_no_trimat()    : opts(flag_no_trimat   ) {} };
  struct opts_no_sympd     : public opts { inline opts_no_sympd()     : opts(flag_no_sympd    ) {} };
  
  static const opts_none         none;
  static const opts_fast         fast;
  static const opts_triu         triu;
  static const opts_tril         tril;
  static const opts_likely_sympd likely_sympd;
  static const opts_no_trimat    no_trimat;
  static const opts_no_sympd     no_sympd;
  }



//! @}
