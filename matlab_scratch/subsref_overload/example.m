t = SubsrefPrinter(2);

t
t();
t(1);
t(:);
t(end, end, end)
t(1:3, :);
t.Hello;
t.('World');
t(5).Hello(20);
% It will even override the current fields
t.Value;

-t

%{
t = 

  SubsrefPrinter with properties:

    Value: 2

subsref:
  type: ()
  subs: []
  
subsref:
  type: ()
  subs: [1.0]
  
subsref:
  type: ()
  subs: [':']
  
subsref:
  type: ()
  subs:
  - [1.0, 2.0, 3.0]
  - ':'
  
subsref:
  {type: ., subs: Hello}
  
subsref:
  {type: ., subs: World}
  
subsref:
  - type: ()
    subs: [5.0]
  - {type: ., subs: Hello}
  - type: ()
    subs: [20.0]
  
subsref:
  {type: ., subs: Value}
%}
