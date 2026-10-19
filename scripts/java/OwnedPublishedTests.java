package slm.owned;

import java.util.*;
import org.junit.runner.Description;
import org.junit.runner.notification.Failure;
import org.junit.runner.notification.RunListener;
import org.junit.runner.notification.RunNotifier;
import org.junit.runners.BlockJUnit4ClassRunner;
import org.junit.runners.model.FrameworkMethod;
import org.junit.platform.engine.TestExecutionResult;
import org.junit.platform.engine.discovery.DiscoverySelectors;
import org.junit.platform.launcher.*;
import org.junit.platform.launcher.core.LauncherDiscoveryRequestBuilder;
import org.junit.platform.launcher.core.LauncherFactory;

/** Retain original tests, enabling teaching-track disabled cases at runner level. */
public final class OwnedPublishedTests {
    private final List<String> registered = new ArrayList<>();
    private final List<String> started = new ArrayList<>();
    private final List<String> passed = new ArrayList<>();
    private final List<String> failed = new ArrayList<>();
    private final List<String> skipped = new ArrayList<>();
    private final List<String> errors = new ArrayList<>();

    private static void failureDetail(Throwable t) {
        for (int i=0; t!=null && i<4; i++, t=t.getCause()) {
            String message=t.toString();
            System.err.println("TEST_FAILURE: " + message.substring(0, Math.min(message.length(), 2000)));
        }
    }

    private static String quote(String s) {
        StringBuilder b = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            if (c == '\\' || c == '"') b.append('\\').append(c);
            else if (c < 32) b.append(String.format("\\u%04x", (int)c));
            else b.append(c);
        }
        return b.append('"').toString();
    }

    private static String array(List<String> items) {
        List<String> encoded = new ArrayList<>();
        for (String s : items) encoded.add(quote(s));
        return "[" + String.join(",", encoded) + "]";
    }

    private static String legacyId(Description d) {
        return d.getClassName() + "#" + d.getMethodName();
    }

    private void registerLegacy(Description d) {
        if (d.isTest()) registered.add(legacyId(d));
        else for (Description child : d.getChildren()) registerLegacy(child);
    }

    private void legacy(Class<?> testClass) throws Exception {
        BlockJUnit4ClassRunner runner = new BlockJUnit4ClassRunner(testClass) {
            @Override protected boolean isIgnored(FrameworkMethod method) { return false; }
        };
        registerLegacy(runner.getDescription());
        RunNotifier notifier = new RunNotifier();
        Set<String> failures = new HashSet<>();
        notifier.addListener(new RunListener() {
            @Override public void testStarted(Description d) { started.add(legacyId(d)); }
            @Override public void testFailure(Failure f) {
                String id = legacyId(f.getDescription()); failures.add(id); failed.add(id);
                failureDetail(f.getException());
            }
            @Override public void testAssumptionFailure(Failure f) {
                String id = legacyId(f.getDescription()); failures.add(id); skipped.add(id);
            }
            @Override public void testIgnored(Description d) { skipped.add(legacyId(d)); }
            @Override public void testFinished(Description d) {
                if (!failures.contains(legacyId(d))) passed.add(legacyId(d));
            }
        });
        runner.run(notifier);
    }

    private void jupiter(List<Class<?>> testClasses) {
        List<org.junit.platform.engine.DiscoverySelector> selectors=new ArrayList<>();
        for (Class<?> testClass:testClasses) selectors.add(DiscoverySelectors.selectClass(testClass));
        LauncherDiscoveryRequest request = LauncherDiscoveryRequestBuilder.request()
            .selectors(selectors)
            .filters(EngineFilter.includeEngines("junit-jupiter"))
            .configurationParameter("junit.jupiter.conditions.deactivate", "org.junit.jupiter.engine.extension.DisabledCondition")
            .configurationParameter("junit.jupiter.execution.parallel.enabled", "false")
            .configurationParameter("junit.jupiter.extensions.autodetection.enabled", "false")
            .build();
        Launcher launcher = LauncherFactory.create();
        TestPlan plan = launcher.discover(request);
        for (TestIdentifier root : plan.getRoots())
            for (TestIdentifier id : plan.getDescendants(root))
                if (id.isTest()) registered.add(id.getUniqueId());
        launcher.execute(plan, new TestExecutionListener() {
            @Override public void dynamicTestRegistered(TestIdentifier id) {
                if (id.isTest()) registered.add(id.getUniqueId());
            }
            @Override public void executionStarted(TestIdentifier id) {
                if (id.isTest()) started.add(id.getUniqueId());
            }
            @Override public void executionSkipped(TestIdentifier id, String reason) {
                if (id.isTest()) skipped.add(id.getUniqueId());
                else errors.add("Skipped container: " + id.getUniqueId());
            }
            @Override public void executionFinished(TestIdentifier id, TestExecutionResult result) {
                if (!id.isTest()) {
                    if (result.getStatus() != TestExecutionResult.Status.SUCCESSFUL)
                        errors.add("Failed container: " + id.getUniqueId());
                    return;
                }
                if (result.getStatus() == TestExecutionResult.Status.SUCCESSFUL) passed.add(id.getUniqueId());
                else if (result.getStatus() == TestExecutionResult.Status.ABORTED) skipped.add(id.getUniqueId());
                else {
                    failed.add(id.getUniqueId());
                    result.getThrowable().ifPresent(OwnedPublishedTests::failureDetail);
                }
            }
        });
    }

    public static void main(String[] args) {
        OwnedPublishedTests run = new OwnedPublishedTests();
        String mode = args[0];
        try {
            List<Class<?>> testClasses=new ArrayList<>();
            for (int i=1;i<args.length;i++)
                testClasses.add(Class.forName(args[i], false, OwnedPublishedTests.class.getClassLoader()));
            if (mode.equals("junit4")) for (Class<?> testClass:testClasses) run.legacy(testClass);
            else if (mode.equals("jupiter")) run.jupiter(testClasses);
            else throw new IllegalArgumentException("Unknown original test framework");
        } catch (Throwable e) { run.errors.add(e.getClass().getName() + ": " + String.valueOf(e.getMessage())); }
        System.out.println("SLM_JAVA_INVENTORY\t{\"mode\":" + quote(mode)
            + ",\"junit4_version\":" + quote(junit.runner.Version.id())
            + ",\"registered\":" + array(run.registered) + ",\"started\":" + array(run.started)
            + ",\"passed\":" + array(run.passed) + ",\"failed\":" + array(run.failed)
            + ",\"skipped\":" + array(run.skipped) + ",\"container_errors\":" + array(run.errors) + "}");
    }
}
