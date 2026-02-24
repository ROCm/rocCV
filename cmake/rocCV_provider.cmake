macro(roccv_provide_dependency method package_name)
    if("${package_name}" STREQUAL "roccv")
        if(TARGET roccv::roccv)
            set(roccv_FOUND TRUE)
        endif()
    endif()
endmacro()

cmake_language(SET_DEPENDENCY_PROVIDER roccv_provide_dependency
    SUPPORTED_METHODS FIND_PACKAGE
)