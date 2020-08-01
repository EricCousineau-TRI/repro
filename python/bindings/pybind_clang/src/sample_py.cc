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

namespace sample {
namespace {

namespace py = pybind11;
using py_rvp = py::return_value_policy;

static llvm::cl::OptionCategory TestCategory("Test");


// https://clang.llvm.org/docs/RAVFrontendAction.html
// https://clang.llvm.org/docs/LibTooling.html#putting-it-together-the-first-tool

class PyASTVisitor : public clang::RecursiveASTVisitor<PyASTVisitor> {
 public:
  PyASTVisitor(py::object cls) : cls_(cls) {}
 
 private:
  py::object cls_;
};

class PyASTConsumer : public clang::ASTConsumer {
 public:
  PyASTConsumer(py::object cls) : visitor_(cls) {}

  void HandleTranslationUnit(clang::ASTContext &Context) override {
    visitor_.TraverseDecl(Context.getTranslationUnitDecl());
  }

 public:
  PyASTVisitor visitor_;
};

class PyASTFrontendAction : public clang::ASTFrontendAction {
 public:
  PyASTFrontendAction(py::object cls) : cls_(cls) {}

  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance& /* ci */,
      clang::StringRef /* file */) {
    return std::make_unique<PyASTConsumer>(cls_);
  }
 private:
  py::object cls_;
};

class PyASTFrontendActionFactory
    : public clang::tooling::FrontendActionFactory {
  public:
    PyASTFrontendActionFactory(py::object cls) : cls_(cls) {}

    clang::FrontendAction* create() override {
      return new PyASTFrontendAction(cls_);
    }
 private:
  py::object cls_;
};

PYBIND11_MODULE(sample, m) {
  {
    // See: clang_indexSourceFile_Impl
    // Should also see: ASTPrinter, CursorVisitor

    using Class = clang::tooling::CommonOptionsParser;
    py::class_<Class>(m, "CommonOptionsParser")
      .def(py::init(
        [](py::list py_argv) {
          // convert :(
          auto cc_argv = py_argv.cast<std::vector<std::string>>();
          int c_argc = cc_argv.size();
          const char** c_argv = new const char*[c_argc];
          for (int i = 0; i < c_argc; ++i) {
            c_argv[i] = cc_argv[i].c_str();
          }
          return new Class(c_argc, c_argv, TestCategory);
        }))
      .def("getCompilations", &Class::getCompilations, py_rvp::reference)
      .def("getSourcePathList", &Class::getSourcePathList);
  }
  {
    using Class = clang::tooling::CompilationDatabase;
    py::class_<Class>(m, "CompilationDatabase");
  }

  {
    using Class = clang::tooling::ClangTool;
    py::class_<Class>(m, "ClangTool")
      .def(py::init(
          [](const clang::tooling::CompilationDatabase& db,
            std::vector<std::string> paths) {
            return new Class(db, paths);
          }))
    .def("run",
        [](Class& self, py::object cls) {
          return self.run(new PyASTFrontendActionFactory(cls));
        });
  }
}

}  // namespace
}  // namespace sample
