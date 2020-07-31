#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Comment.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace test {
namespace {

namespace py = pybind11;

static llvm::cl::OptionCategory TestCategory("Test");


class PyASTConsumer : public clang::ASTConsumer {
};

class PyASTFrontendAction : public clang::ASTFrontendAction {
 public:
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance &ci, clang::StringRef file) {
    return std::make_unique<PyASTConsumer>(cls(), &ci);
  }
 private:
  py::object cls;
};

PYBIND11_MODULE(test, m) {
  {
    // See: clang_indexSourceFile_Impl

    using Class = clang::CommonOptionsParser;
    py::class_<>(m, "CommonOptionParser")
        .def(py::init(
          [](py::list py_argv) {
            auto cc_argv = py_argv.cast<std::vector<std::string>>();
            int argc = cc_argv.size();
            char** c_argv = new char*[argc];
            for (int i = 0; i < argc; ++i) {
              c_argv[i] = cc_argv[i].c_str();
            }
            return new Class(argc, argv, TestCategory);
          }))
        .def("getCompilations", &Class::getCompilations)
        .def("getSourcePathList", &Class::getSourcePathList);
  }
  py::class_<clang::CompilationDatabase>(m, "CompilationDatabase");

  py::class_<clang::ClangTool>(m, "ClangTool")
      .def(py::init(
          [](const CompilationDatabase& db, std::vector<std::string> paths) {
            return new ClangTool(db, paths);
          }))
      .def("run",
          [](py::object cls) {
          });

  CommonOptionsParser op(argc, argv, TestCategory);
  ClangTool tool(op.getCompilations(), op.getSourcePathList());
  return tool.run(newFrontendActionFactory<BinderFrontendAction>().get());
}

}  // namespace
}  // namespace test
