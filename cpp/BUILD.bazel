package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "copy_ctor",
    srcs = ["copy_ctor.cc"],
)

cc_library(
    name = "name_trait",
    hdrs = ["name_trait.h"],
)

cc_binary(
    name = "generator",
    srcs = ["generator.cc"],
)

cc_binary(
    name = "generator_godbolt",
    srcs = ["generator_godbolt.cc"],
)

cc_binary(
    name = "tpl_inst_main",
    srcs = ["tpl_inst_main.cc", "tpl_inst.cc", "tpl_inst.h", "name_trait.h"],
)

cc_binary(
    name = "virtual_template",
    srcs = ["virtual_template.cc"],
)

cc_binary(
    name = "forward",
    srcs = ["forward.cc"],
)

cc_binary(
    name = "recursive_template",
    srcs = ["recursive_template.cc"],
)

cc_binary(
    name = "constexpr",
    srcs = ["constexpr.cc"],
)

cc_binary(
    name = "name_trait_main",
    srcs = ["name_trait_main.cc", "name_trait.h"],
)

cc_binary(
    name = "prevent_implicit_conversion",
    srcs = ["prevent_implicit_conversion.cc", "name_trait.h", "drake_copy.h"],
)

cc_binary(
    name = "tpl_spec_return_type_main",
    srcs = ["tpl_spec_return_type_main.cc", "tpl_spec_return_type.cc", "tpl_spec_return_type.h", "name_trait.h"],
)

cc_binary(
    name = "auto_inference",
    srcs = ["auto_inference.cc", "name_trait.h"],
)

cc_binary(
    name = "tpl_variadic_final",
    srcs = ["tpl_variadic_final.cc", "name_trait.h"],
)

cc_binary(
    name = "tuple",
    srcs = ["tuple.cc", "name_trait.h"],
)

cc_binary(
    name = "tuple_future_main",
    srcs = ["tuple_future_main.cc", "tuple_future.h"],
)

cc_binary(
    name = "tpl_spec_greedy",
    srcs = ["tpl_spec_greedy.cc", "name_trait.h"],
)

cc_binary(
    name = "overload_options",
    srcs = ["overload_options.cc"],
)

cc_binary(
    name = "tpl_check_errors",
    srcs = ["tpl_check_errors.cc", "name_trait.h"],
)

cc_binary(
    name = "composition_ctor",
    srcs = ["composition_ctor.cc", "name_trait.h"],
)

cc_binary(
    name = "tpl_ref_greedy",
    srcs = ["tpl_ref_greedy.cc", "name_trait.h"],
)

cc_binary(
    name = "tpl_tpl_forward",
    srcs = ["tpl_tpl_forward.cc", "name_trait.h"],
)

cc_binary(
    name = "tpl_ref",
    srcs = ["tpl_ref.cc", "name_trait.h"],
)

cc_binary(
    name = "rvalue_check",
    srcs = ["rvalue_check.cc", "name_trait.h"],
)

cc_binary(
    name = "tuple_iter",
    srcs = ["tuple_iter.cc", "tuple_iter.h", "name_trait.h", "tuple_future.h"],
)

cc_binary(
    name = "make_shared_alias",
    srcs = ["make_shared_alias.cc", "name_trait.h"],
)

cc_binary(
    name = "tpl_friend",
    srcs = ["tpl_friend.cc"],
)

cc_binary(
    name = "tpl_method_overload",
    srcs = ["tpl_method_overload.cc", "name_trait.h", "drake_copy.h"],
)

cc_binary(
    name = "lifetime",
    srcs = ["lifetime.cc"],
)

cc_binary(
    name = "benchmark_map_test",
    srcs = ["benchmark_map_test.cc"],
    deps = ["//externals/benchmark"],
)

cc_binary(
    name = "check_for_type",
    srcs = ["check_for_type.cc", "name_trait.h"],
)

cc_binary(
    name = "tpl_tpl_pod",
    srcs = ["tpl_tpl_pod.cc"],
)

cc_binary(
    name = "tpl_spec_switch_main",
    srcs = ["tpl_spec_switch_main.cc", "tpl_spec_switch.cc", "tpl_spec_switch.h"],
)

cc_binary(
    name = "auto_flexibility",
    srcs = ["auto_flexibility.cc", "name_trait.h"],
)

cc_binary(
    name = "tpl_rref_only",
    srcs = ["tpl_rref_only.cc"],
    deps = [":name_trait"],
)

cc_binary(
    name = "optional",
    srcs = ["optional.cc"],
)

cc_binary(
    name = "tpl_pack_visitor",
    srcs = ["tpl_pack_visitor.cc"],
)

cc_binary(
    name = "c_style_init",
    srcs = ["c_style_init.cc"],
)

cc_binary(
    name = "tpl_pack_sizing",
    srcs = ["tpl_pack_sizing.cc"],
)

cc_binary(
    name = "friend_tpl",
    srcs = ["friend_tpl.cc"],
)

cc_binary(
    name = "stl_ref",
    srcs = ["stl_ref.cc"],
)

cc_binary(
    name = "stl_container",
    srcs = [
        "stl_container.cc",
    ],
)

cc_binary(
    name = "stl_pod",
    srcs = [
        "stl_pod.cc",
    ],
)

cc_binary(
    name = "tpl_impl",
    srcs = ["tpl_impl.cc"],
)

cc_binary(
    name = "gtest_op",
    srcs = ["gtest_op.cc"],
    deps = [
        "@gtest//:main",
    ],
    testonly = 1,
)

cc_binary(
    name = "shared_ptr_release",
    srcs = ["shared_ptr_release.cc"],
)

cc_binary(
    name = "lambda_stack_capture",
    srcs = ["lambda_stack_capture.cc"],
)

cc_binary(
    name = "raw_string",
    srcs = ["raw_string.cc"],
)

cc_binary(
    name = "func_target",
    srcs = ["func_target.cc"],
)

cc_binary(
    name = "flags_special",
    srcs = ["flags_special.cc"],
)

cc_binary(
    name = "constexpr_class",
    srcs = ["constexpr_class.cc"],
)

cc_library(
    name = "simple_converter",
    srcs = ["simple_converter.h"],
    deps = [":type_pack"],
)

cc_binary(
    name = "simple_converter_test",
    srcs = ["simple_converter_test.cc"],
    deps = [
        ":simple_converter",
        ":name_trait",
    ],
)

cc_binary(
    name = "unique_ptr_stuff",
    srcs = ["unique_ptr_stuff.cc"],
)

cc_binary(
    name = "shared_ptr_stuff",
    srcs = ["shared_ptr_stuff.cc"],
)

cc_binary(
    name = "cast",
    srcs = ["cast.cc"],
)

cc_binary(
    name = "alias_test",
    srcs = ["alias_test.cc"],
)

cc_library(
    name = "type_pack",
    hdrs = ["type_pack.h"],
)

cc_binary(
    name = "tpl_pack_visitor_ext",
    srcs = ["tpl_pack_visitor_ext.cc"],
    deps = [":type_pack"],
)

cc_binary(
    name = "void_check",
    srcs = ["void_check.cc"],
)

cc_binary(
    name = "tpl_literal",
    srcs = ["tpl_literal.cc"],
    deps = [":name_trait"],
)

cc_binary(
    name = "alias_dup_test",
    srcs = ["alias_dup_test.cc"],
)

cc_binary(
    name = "tpl_perfect_fwd",
    srcs = ["tpl_perfect_fwd.cc"],
    deps = [":name_trait"],
)

cc_binary(
    name = "unique_ptr_tracked",
    srcs = ["unique_ptr_tracked.cc"],
)

cc_binary(
    name = "type_erased",
    srcs = ["type_erased.cc"],
)

cc_binary(
    name = "tpl_tag",
    srcs = ["tpl_tag.cc"],
)

cc_binary(
    name = "tpl_crtp_spec",
    srcs = ["tpl_crtp_spec.cc"],
    deps = [":name_trait"],
)

cc_library(
    name = "wrap_function",
    hdrs = ["wrap_function.h"],
    deps = [":type_pack"],
)

cc_binary(
    name = "wrap_function_main",
    srcs = ["wrap_function_main.cc"],
    deps = [
        ":name_trait",
        ":wrap_function",
    ],
)

cc_binary(
    name = "thread_daemon",
    srcs = ["thread_daemon.cc"],
    linkopts = ["-lpthread"],
)

cc_binary(
    name = "clone_stuff",
    srcs = ["clone_stuff.cc"],
)

cc_binary(
    name = "friend_stuff",
    srcs = ["friend_stuff.cc"],
)

cc_binary(
    name = "func_ptr_c",
    srcs = ["func_ptr_c.cc"],
)

cc_binary(
    name = "type_pack_stuff",
    srcs = ["type_pack_stuff.cc"],
    deps = [":type_pack", ":name_trait"],
)

cc_binary(
    name = "enum_ostream_overload",
    srcs = ["enum_ostream_overload.cc"],
)

cc_binary(
    name = "optional_stuff",
    srcs = ["optional_stuff.cc"],
)

cc_binary(
    name = "unique_ptr_cast",
    srcs = ["unique_ptr_cast.cc"],
)

cc_binary(
    name = "enum_int",
    srcs = ["enum_int.cc"],
)

cc_binary(
    name = "doc_struct_nest",
    srcs = [
        "doc_struct_nest.cc",
        "doc_struct_nest_big.h",
    ],
)

cc_binary(
    name = "move_check",
    srcs = ["move_check.cc"],
)

cc_binary(
    name = "scalar_functor_set",
    srcs = ["scalar_functor_set.cc"],
    copts = ["-std=c++14"],
)

cc_binary(
    name = "lambda_recursive",
    srcs = ["lambda_recursive.cc"],
    copts = ["-std=c++14"],
)

cc_binary(
    name = "function_or_method",
    srcs = ["function_or_method.cc"],
    copts = ["-std=c++14"],
)
