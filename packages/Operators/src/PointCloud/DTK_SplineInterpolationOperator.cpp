//---------------------------------------------------------------------------//
/*
  Copyright (c) 2012, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the University of Wisconsin - Madison nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//---------------------------------------------------------------------------//
/*!
 * \file   DTK_SplineInterpolationOperator.cpp
 * \author Stuart R. Slattery
 * \brief Parallel spline interpolator.
 */
//---------------------------------------------------------------------------//

#include "DTK_SplineInterpolationOperator_impl.hpp"

#include "DTK_BuhmannBasis.hpp"
#include "DTK_WendlandBasis.hpp"
#include "DTK_WuBasis.hpp"

namespace DataTransferKit
{
template class SplineInterpolationOperator<WendlandBasis<0>, 1>;
template class SplineInterpolationOperator<WendlandBasis<2>, 1>;
template class SplineInterpolationOperator<WendlandBasis<4>, 1>;
template class SplineInterpolationOperator<WendlandBasis<6>, 1>;
template class SplineInterpolationOperator<WendlandBasis<21>, 1>;

template class SplineInterpolationOperator<WendlandBasis<0>, 2>;
template class SplineInterpolationOperator<WendlandBasis<2>, 2>;
template class SplineInterpolationOperator<WendlandBasis<4>, 2>;
template class SplineInterpolationOperator<WendlandBasis<6>, 2>;
template class SplineInterpolationOperator<WendlandBasis<21>, 2>;

template class SplineInterpolationOperator<WendlandBasis<0>, 3>;
template class SplineInterpolationOperator<WendlandBasis<2>, 3>;
template class SplineInterpolationOperator<WendlandBasis<4>, 3>;
template class SplineInterpolationOperator<WendlandBasis<6>, 3>;
template class SplineInterpolationOperator<WendlandBasis<21>, 3>;

template class SplineInterpolationOperator<WuBasis<2>, 1>;
template class SplineInterpolationOperator<WuBasis<4>, 1>;

template class SplineInterpolationOperator<WuBasis<2>, 2>;
template class SplineInterpolationOperator<WuBasis<4>, 2>;

template class SplineInterpolationOperator<WuBasis<2>, 3>;
template class SplineInterpolationOperator<WuBasis<4>, 3>;

template class SplineInterpolationOperator<BuhmannBasis<3>, 1>;

template class SplineInterpolationOperator<BuhmannBasis<3>, 2>;

template class SplineInterpolationOperator<BuhmannBasis<3>, 3>;

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//
// end DTK_SplineInterpolationOperator.cpp
//---------------------------------------------------------------------------//
