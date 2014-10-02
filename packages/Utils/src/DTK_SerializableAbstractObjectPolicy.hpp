//---------------------------------------------------------------------------//
/*
  Copyright (c) 2014, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the Oak Ridge National Laboratory nor the
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
 * \brief DTK_SerializableAbstractObjectPolicy.hpp
 * \author Stuart R. Slattery
 * \brief Serializable abstract object policy interface.
 */
//---------------------------------------------------------------------------//

#ifndef DTK_SERIALIZABLEABSTRACTOBJECTPOLICY_HPP
#define DTK_SERIALIZABLEABSTRACTOBJECTPOLICY_HPP

#include <string>

#include "DTK_DBC.hpp"
#include "DTK_AbstractBuilder.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayView.hpp>

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
/*!
  \class UndefinedSerializableAbstractObjectPolicy
  \brief Complie time error indicator for policy implementations.
*/
//---------------------------------------------------------------------------//
template<typename T>
struct UndefinedSerializableAbstractObjectPolicy 
{
    static inline T notDefined() 
    {
	return T::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
  \class SerializableAbstractObjectPolicy
  \brief Interface definition for objects that can be serialized.

  This class provides a runtime mechanism to serialize a derived class of
  arbitrary size through a base class interface and deserialize the base class
  with the correct underlying derived class.
*/
//---------------------------------------------------------------------------//
template<class T>
class SerializableAbstractObjectPolicy
{
  public:

    //! Base class type.
    typedef T object_type;

    //@{
    //! Identification functions.
    /*!
     * \brief Return a string indicating the derived object type.
     * \return A string indicating the type of derived object implementing the
     * interface. This string will drive object construction with the builder.
     */
    static std::string objectType( const Teuchos::RCP<T>& object )
    {
	UndefinedSerializableAbstractObjectPolicy<T>::notDefined();
	return std::string("Not implemented");
    }
    //@}

    //@{
    //! Serialization functions.
    /*
     * \brief Get the size of the serialized object in bytes.
     * \return The size of the object when serialized in bytes.
     */
    static std::size_t byteSize( const Teuchos::RCP<T>& object )
    {
	UndefinedSerializableAbstractObjectPolicy<T>::notDefined();
	return 0;
    }

    /*
     * \brief Serialize the object into a buffer.
     * \param buffer A view into a data buffer of size byteSize(). Write the
     * serialized object into this view.
     */
    static void serialize( const Teuchos::RCP<T>& object,
			   const Teuchos::ArrayView<char>& buffer )
    {
	UndefinedSerializableAbstractObjectPolicy<T>::notDefined();
    }

    /*!
     * \brief Deserialize an object from a buffer.
     * \param buffer A view into a data buffer of size byteSize(). Deserialize
     * the object from this view.
     */
    static void deserialize( const Teuchos::RCP<T>& object,
			     const Teuchos::ArrayView<const char>& buffer )
    {
	UndefinedSerializableAbstractObjectPolicy<T>::notDefined();
    }

    //@}

    //@{
    //! Polymorphic construction functions.
    /*!
     * \brief Static function for getting the builder for the base class.
     */
    static Teuchos::RCP<AbstractBuilder<T> > getBuilder()
    {
	UndefinedSerializableAbstractObjectPolicy<T>::notDefined();
	return Teuchos::null;
    }
    //@}
};

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//

#endif // end DTK_SERIALIZABLEABSTRACTOBJECTPOLICY_HPP

//---------------------------------------------------------------------------//
// end DTK_SerializableAbstractObjectPolicy.hpp
//---------------------------------------------------------------------------//
