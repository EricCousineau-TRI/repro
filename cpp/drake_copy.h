#pragma once

/** DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN defaults the special member
functions for copy-construction, copy-assignment, move-construction, and
move-assignment.  This macro should be used only when copy-construction and
copy-assignment defaults are well-formed.  Note that the defaulted move
functions could conceivably still be ill-formed, in which case they will
effectively not be declared or used -- but because the copy constructor exists
the type will still be MoveConstructible.  Drake's Doxygen is customized to
render the functions in detail, with appropriate comments.  Invoke this this
macro in the public section of the class declaration, e.g.:
<pre>
class Foo {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Foo)

  // ...
};
</pre>
*/
#define DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Classname)       \
  Classname(const Classname&) = default;                        \
  Classname& operator=(const Classname&) = default;             \
  Classname(Classname&&) = default;                             \
  Classname& operator=(Classname&&) = default;                  \
  /* Fails at compile-time if default-copy doesn't work. */     \
  static void DRAKE_COPYABLE_DEMAND_COPY_CAN_COMPILE() {        \
    (void) static_cast<Classname& (Classname::*)(               \
        const Classname&)>(&Classname::operator=);              \
  }
